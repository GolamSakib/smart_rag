import faiss
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query,Request,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List, Optional
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from uuid import uuid4
from collections import defaultdict
import mysql.connector
from mysql.connector import errorcode
import os
import subprocess
import decimal
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import httpx
load_dotenv()  # Load .env file

# Environment variables (create a .env file in your project root)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_verify_token")  # From Step 2
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")  # From Step 2
FB_GRAPH_URL = f"https://graph.facebook.com/v21.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"  # v18.0 is current as of 2025; check docs for updates

app = FastAPI()

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for per-session memory
session_memories = defaultdict(lambda: {
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "last_products": []  # Store last retrieved products
})

# Load CLIP model for image embeddings
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Load FAISS indexes
image_index = faiss.read_index('vector_stores/image_faiss/image.index')
text_vector_store = LangchainFAISS.load_local(
    'vector_stores/text_faiss', embeddings, allow_dangerous_deserialization=True
)

# Load image metadata
with open('vector_stores/image_faiss/image_metadata.json', 'r') as f:
    image_metadata = json.load(f)

# Configure OpenAI-compatible LLM (Ollama API server)
llm = ChatOpenAI(
    # model="mistral:latest",
    model="openai/gpt-oss-20b:free",
    # base_url="http://localhost:11434/v1",
    base_url="https://openrouter.ai/api/v1",
    # api_key='ollama',
    api_key='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',
    max_tokens=300,
    temperature=0.7
)

# Updated Prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "You are a helpful sales assistant. ALWAYS respond in Bengali, using a polite, natural, and persuasive conversational tone.\n"
        "When mentioning product details (name, description, price), preserve them exactly as they appear in the context without translation.\n"
        "Use the context and chat history to answer the user's query.\n"
        "If the user uploads an image or asks about a product, provide the product name, description, and price (exclude marginal price).\n"
        "If the user asks 'pp' or similar (case-insensitive), respond only with the price of the most relevant product from the context and the price should be always in taka.\n"
        "If the user asks whether the product looks exactly like the image (e.g., 'hubohu chobir moto'), respond persuasively with: "
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা নিশ্চিত করি যে আপনি ছবিতে যা দেখছেন, ঠিক তেমনটাই পাবেন।'\n"
        "If the user wants to order, reply with:\n"
        "'অনুগ্রহ করে আপনার অর্ডার সম্পূর্ণ করতে নিচের তথ্য দিন:\nআপনার নাম:\nআপনার ঠিকানা:\nআপনার ফোন নাম্বার:'\n"
        "If the user asks to bargain, use the marginal price to offer a discount but never below marginal price.\n"
        "If asked about delivery, say in Bengali: 'আপনি যদি ঢাকায় থাকেন তবে ১ দিনের মধ্যে পণ্য পাবেন, অন্যথায় ২ দিনের মধ্যে।'\n"
        "Do not mention marginal price unless asked.\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User: {user_query}\nBot:"
    )
)

# Chain
chain = RunnableSequence(prompt | llm)

# Image embedding function
def get_image_embedding(image: Image.Image):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = image_model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()

# Database Configuration
DB_CONFIG = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'smart_rag',
}

# Database Connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return None

# API Models
class Product(BaseModel):
    name: str
    description: str
    price: float
    code: str
    marginal_price: float
    image_ids: List[int]

# API Endpoints
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    print(f"VERIFY_TOKEN: {os.getenv('VERIFY_TOKEN')}, Received token: {token}")  # Debug
    
    if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN", "heaven"):
        return Response(content=str(challenge), status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()
    if data.get("object") != "page":
        return JSONResponse(status_code=200, content={"status": "ok"})

    for entry in data.get("entry", []):
        messaging = entry.get("messaging", [])
        for message_data in messaging:
            sender_id = message_data["sender"]["id"]
            print("Received message_data:", message_data)  # Debug log
            incoming_msg = message_data["message"].get("text", "")
            files = []

            # Handle attachments
            if "attachments" in message_data["message"]:
                attachment = message_data["message"]["attachments"][0]
                if attachment["type"] == "image":
                    image_url = attachment["payload"]["url"]
                    async with httpx.AsyncClient() as client:
                        image_response = await client.get(image_url)
                        if image_response.status_code == 200:
                            image_content = image_response.content
                            files = [("images", (f"image_{sender_id}.jpg", image_content, "image/jpeg"))]
                        else:
                            print(f"Failed to download image: {image_response.status_code}")
                            send_to_facebook(sender_id, "Sorry, I couldn't process the image.")
                            continue

            # Check for empty input
            if not incoming_msg and not files:
                send_to_facebook(sender_id, "Please send a text message or an image to search for products.")
                continue

            session_id = sender_id
            async with httpx.AsyncClient() as client:
                print("Sending to /chat:", {"text": incoming_msg, "session_id": session_id, "files": bool(files)})
                if files:
                    response = await client.post(
                        "https://b0e59f6cdaea.ngrok-free.app/chat",
                        data={"text": incoming_msg, "session_id": session_id},
                        files=files,
                        timeout=30.0
                    )
                else:
                    response = await client.post(
                        "https://b0e59f6cdaea.ngrok-free.app/chat",
                        data={"text": incoming_msg, "session_id": session_id},
                        timeout=30.0
                    )

                if response.status_code != 200:
                    print(f"Error from /chat: {response.status_code}, {response.text}")
                    bot_reply = "Sorry, something went wrong."
                else:
                    result = response.json()
                    bot_reply = result["reply"]

            send_to_facebook(sender_id, bot_reply)

    return JSONResponse(status_code=200, content={"status": "ok"})

def send_to_facebook(recipient_id: str, message_text: str):
    """Send message back to user via Facebook Graph API."""
    payload = {
        "messaging_type": "RESPONSE",
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    response = requests.post(
        FB_GRAPH_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        print(f"Error sending message: {response.text}")

# Chat Endpoint
@app.post("/chat")
async def chat(
    images: Optional[List[UploadFile]] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    if not images and not text:
        return JSONResponse(status_code=400, content={"error": "At least one image or text input is required"})

    session_id = session_id or str(uuid4())
    session_data = session_memories[session_id]
    memory = session_data["memory"]
    retrieved_products = session_data["last_products"]

    # Image search
    if images:
        retrieved_products = []
        for image_file in images:
            image = Image.open(image_file.file)
            image_embedding = get_image_embedding(image)
            D, I = image_index.search(np.array([image_embedding]).astype('float32'), k=1)
            retrieved_products.append(image_metadata[I[0][0]])
        session_data["last_products"] = retrieved_products

    # Text search
    if text:
        docs = text_vector_store.similarity_search(text, k=3)
        for doc in docs:
            retrieved_products.append(doc.metadata)
        session_data["last_products"] = retrieved_products

    # Remove duplicates
    seen_products = set()
    unique_products = []
    for product in retrieved_products:
        # Create a unique identifier for each product based on its name and code
        identifier = (product.get('name', '').strip(), product.get('code', '').strip())
        if identifier not in seen_products:
            seen_products.add(identifier)
            unique_products.append(product)
    retrieved_products = unique_products
    

    # Build context
    context = "\nAvailable products:\n"
    for product in retrieved_products:
        context += f"- Name: {product['name']}, Description: {product['description']}, Price: {product['price']}, Marginal Price: {product['marginal_price']}\n"
    

    # Define query
    user_query = text.strip() if text else "Provide the name, description, and price of the product in the uploaded image."

    # if any(k in user_query.lower() for k in ["order", "buy", "purchase", "confirm", "অর্ডার", "ক্রয়", "কিনতে", "কিনবো", "অর্ডার করতে", "অর্ডার দিন", "কনফার্ম"]):
    #     bot_response = "অনুগ্রহ করে আপনার অর্ডার সম্পূর্ণ করতে নিচের তথ্য দিন:\nআপনার নাম:\nআপনার ঠিকানা:\nআপনার ফোন নাম্বার:"
    # elif any(k in user_query.lower() for k in ["delivery", "deliver", "shipping", "ship", "how many days", "when will i get", "how long", "koydin"]):
    #     bot_response = "আপনি যদি ঢাকায় থাকেন তবে ১ দিনের মধ্যে পণ্য পাবেন, অন্যথায় ২ দিনের মধ্যে।"
    if any(k in user_query.lower() for k in ["hubohu", "exactly like", "same as picture", "ছবির মত", "হুবহু"]):
        bot_response = "হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা নিশ্চিত করি যে আপনি ছবিতে যা দেখছেন, ঠিক তেমনটাই পাবেন।"
    else:
        chat_history = memory.load_memory_variables({})["chat_history"]
        inputs = {"chat_history": chat_history, "user_query": user_query, "context": context}
        response = chain.invoke(inputs)
        bot_response = response.content
    # return chat_history

    # Save to memory
    memory.save_context({"user_query": user_query}, {"output": bot_response})

    return JSONResponse(content={
        "reply": bot_response,
        "related_products": [{k: v for k, v in product.items() if k != "marginal_price"} for product in retrieved_products],
        "session_id": session_id
    })

# Image Management
@app.post("/api/images")
async def upload_images(files: List[UploadFile] = File(...)):
    if not os.path.exists("product-image"):
        os.makedirs("product-image")

    saved_images = []
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor()

    for file in files:
        file_path = os.path.join("product-image", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        add_image = ("INSERT INTO images (image_path) VALUES (%s)")
        data_image = (file_path,)
        cursor.execute(add_image, data_image)
        conn.commit()
        image_id = cursor.lastrowid
        saved_images.append({"id": image_id, "image_path": file_path})

    cursor.close()
    conn.close()

    return JSONResponse(content=saved_images)

@app.get("/api/images")
def get_images():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM images ORDER BY created_at DESC")
    images = cursor.fetchall()
    cursor.close()
    conn.close()
    return images

@app.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT image_path FROM images WHERE id = %s", (image_id,))
    image = cursor.fetchone()
    if image and os.path.exists(image['image_path']):
        os.remove(image['image_path'])

    cursor.execute("DELETE FROM images WHERE id = %s", (image_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Image deleted successfully"}

# Product Management
@app.post("/api/products")
def create_product(product: Product):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor()
    add_product = ("INSERT INTO products "
                   "(name, description, price, code, marginal_price) "
                   "VALUES (%s, %s, %s, %s, %s)")
    data_product = (product.name, product.description, product.price, product.code, product.marginal_price)
    cursor.execute(add_product, data_product)
    conn.commit()
    product_id = cursor.lastrowid

    if product.image_ids:
        add_product_image = ("INSERT INTO product_images (product_id, image_id) VALUES (%s, %s)")
        for image_id in product.image_ids:
            cursor.execute(add_product_image, (product_id, image_id))
        conn.commit()

    cursor.close()
    conn.close()
    return {"id": product_id}

@app.get("/api/products")
def get_products(name: Optional[str] = None, code: Optional[str] = None, min_price: Optional[str] = None, max_price: Optional[str] = None):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT p.*, GROUP_CONCAT(i.id) as image_ids, GROUP_CONCAT(i.image_path) as images FROM products p LEFT JOIN product_images pi ON p.id = pi.product_id LEFT JOIN images i ON pi.image_id = i.id WHERE 1=1"
    params = []

    if name:
        query += " AND p.name LIKE %s"
        params.append(f"%{name}%")
    if code:
        query += " AND p.code = %s"
        params.append(code)
    
    if min_price:
        try:
            min_price_float = float(min_price)
            query += " AND p.price >= %s"
            params.append(min_price_float)
        except (ValueError, TypeError):
            pass
            
    if max_price:
        try:
            max_price_float = float(max_price)
            query += " AND p.price <= %s"
            params.append(max_price_float)
        except (ValueError, TypeError):
            pass

    query += " GROUP BY p.id ORDER BY p.id DESC"
    
    cursor.execute(query, tuple(params))
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    
    for product in products:
        if product['images']:
            product['images'] = product['images'].split(',')
            product['image_ids'] = [int(id) for id in product['image_ids'].split(',')]
        else:
            product['images'] = []
            product['image_ids'] = []
            
    return products

@app.put("/api/products/{product_id}")
def update_product(product_id: int, product: Product):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor()
    
    update_prod = ("UPDATE products SET name=%s, description=%s, price=%s, code=%s, marginal_price=%s WHERE id=%s")
    data_prod = (product.name, product.description, product.price, product.code, product.marginal_price, product_id)
    cursor.execute(update_prod, data_prod)

    cursor.execute("DELETE FROM product_images WHERE product_id = %s", (product_id,))
    if product.image_ids:
        add_product_image = ("INSERT INTO product_images (product_id, image_id) VALUES (%s, %s)")
        for image_id in product.image_ids:
            cursor.execute(add_product_image, (product_id, image_id))
    
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Product updated successfully"}

@app.delete("/api/products/{product_id}")
def delete_product(product_id: int):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM products WHERE id = %s", (product_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Product deleted successfully"}

# JSON Generation and Training
@app.post("/api/generate-json")
def generate_json():
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT p.*, i.image_path FROM products p JOIN product_images pi ON p.id = pi.product_id JOIN images i ON pi.image_id = i.id")
        products = cursor.fetchall()
        cursor.close()
        conn.close()

        for product in products:
            for key, value in product.items():
                if isinstance(value, decimal.Decimal):
                    product[key] = float(value)

        if not os.path.exists("data"):
            os.makedirs("data")

        with open("data/products.json", "w") as f:
            json.dump(products, f, indent=4)
            
        return {"message": "products.json generated successfully"}
    except Exception as e:
        print("Error in generate_json:", str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/train")
def train_model():
    try:
        result = subprocess.run(["python", "training.py"], capture_output=True, text=True, check=True)
        return {"message": "Training completed successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")

# Static Files
app.mount("/product-image", StaticFiles(directory="product-image"), name="product-image")
app.mount("/", StaticFiles(directory="C:\\Users\\USER\\Desktop\\smart_rag", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)