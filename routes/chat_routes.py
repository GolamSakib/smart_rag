from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
from typing import List, Optional
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from uuid import uuid4
from collections import defaultdict
import numpy as np
import httpx
import requests

from services.model_manager import model_manager
from config.settings import settings
from services.database_service import db_service

router = APIRouter()

# In-memory store for per-session memory
session_memories = defaultdict(lambda: {
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "last_products": []  # Store last retrieved products
})

# Updated Prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "You are a friendly and professional sales assistant. Begin your response with the Islamic greeting 'Assalamu Alaikum' to align with Muslim cultural norms, and maintain a polite, natural, and persuasive tone in English or Bengali (as specified) to encourage purchases.\n"
        "Preserve product details (name, description, price) exactly as provided in the context without translation.\n"
        "Use the context and chat history to address the user's query accurately and enticingly.\n"
        "If the user uploads images or asks about a product, include the product name, description, and price (in taka, excluding any reference to marginal price).\n"
        "If the user asks 'pp' or similar (case-insensitive), respond only with the price of the most relevant product from the context in taka.\n"
        "If the user asks if the product matches the image (e.g., 'hubohu chobir moto'), respond persuasively in Bengali: "
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা গ্যারান্টি দিচ্ছি, ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n"
        "If the user wants to order, respond in Bengali to finalize the order: "
        "'📦 অর্ডার কনফার্ম করার জন্য দয়া করে আপনার\n"
        "👤 নাম\n"
        "🏠 ঠিকানা\n"
        "📱 মোবাইল নাম্বারটি দিন।\n"
        "💰 চিন্তার কিছু নেই — আমরা কোনো রকম এডভান্স নেই না।\n"
        "🛍 আপনি প্রোডাক্ট হাতে পাবার পর ভালোভাবে দেখে তবেই টাকা পরিশোধ করবেন (Cash on Delivery)。'\n"
        "If the user asks to bargain (e.g., 'dam komano jay kina', 'ektu komano jay na', 'dam ta onk beshi', or similar phrases), respond persuasively in Bengali, offering a discount without mentioning 'marginal price' or 'margin.' For example: "
        "'আপনার জন্য আমরা বিশেষ ছাড় দিচ্ছি! দামটা একটু কমিয়ে [offer price] টাকা করতে পারি, এর চেয়ে ভালো ডিল পাবেন না! এখনই অর্ডার করলে দ্রুত ডেলিভারি নিশ্চিত।'\n"
        "If asked about delivery, respond in Bengali: "
        "'আমরা সারা বাংলাদেশে \"ফুল ক্যাশ অন\" হোম ডেলিভারি করে থাকি।\n"
        "🏠 সহজ ও নিরাপদ ডেলিভারি: আপনার বাড়িতেই প্রোডাক্ট পৌঁছে যাবে, ঝামেলা ছাড়াই।\n"
        "🚚 দ্রুত ও নির্ভরযোগ্য ডেলিভারি: পাঠাও কুরিয়ারের মাধ্যমে দ্রুত প্রোডাক্ট পৌঁছানো হয়।\n"
        "👀 পণ্য হাতে দেখে চেক করার সুযোগ: পণ্য গ্রহণের সময় ভালো করে পরীক্ষা করে নিতে পারবেন।\n"
        "💳 নিরাপদ পেমেন্ট পদ্ধতি: পেমেন্ট শুধুমাত্র পণ্য গ্রহণের পরই দিতে হবে।\n"
        "📍 ঢাকার মধ্যে: আপনার অর্ডারকৃত পণ্যটি পৌঁছে যাবে ১-২ দিনের মধ্যে।\n"
        "🚚 ঢাকার বাইরে: অর্ডারকৃত পণ্যটি ২-৩ দিনের মধ্যে আপনার ঠিকানায় পৌঁছে যাবে ইনশাআল্লাহ।\n"
        "🎁 আমরা প্রতিটি অর্ডারে ভালোবাসা ও যত্ন দিয়ে ডেলিভারি নিশ্চিত করি।'\n"
        "Never mention 'marginal price,' 'margin,' or 'মার্জিনাল' in any response, even during bargaining or when explicitly asked, to avoid customer confusion. Instead, focus on the offered price and product value.\n"
        "Highlight the product's value and reliability to make the offer irresistible.\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User: {user_query}\nBot: Assalamu Alaikum! "
    )
)


@router.post("/chat")
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
        image_index = model_manager.get_image_index()
        image_metadata = model_manager.get_image_metadata()
        
        if image_index is None or not image_metadata:
            return JSONResponse(status_code=500, content={"error": "Image search not available"})
            
        for image_file in images:
            image = Image.open(image_file.file)
            image_embedding = model_manager.get_image_embedding(image)
            D, I = image_index.search(np.array([image_embedding]).astype('float32'), k=1)
            retrieved_products.append(image_metadata[I[0][0]])
        session_data["last_products"] = retrieved_products

    # Text search
    if text:
        text_vector_store = model_manager.get_text_vector_store()
        if text_vector_store is None:
            return JSONResponse(status_code=500, content={"error": "Text search not available"})
            
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
    print(context)

    # Define query
    user_query = text.strip() if text else "Provide the name, description, and price of the products in the uploaded images."

    if any(k in user_query.lower() for k in ["hubohu", "exactly like", "same as picture", "ছবির মত", "হুবহু"]):
        bot_response = "হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা নিশ্চিত করি যে আপনি ছবিতে যা দেখছেন, ঠিক তেমনটাই পাবেন।"
    else:
        llm = model_manager.get_llm()
        chain = RunnableSequence(prompt | llm)
        chat_history = memory.load_memory_variables({})["chat_history"]
        inputs = {"chat_history": chat_history, "user_query": user_query, "context": context}
        response = chain.invoke(inputs)
        bot_response = response.content

    # Increment message count
    try:
        with db_service.get_cursor() as (cursor, connection):
            cursor.execute("UPDATE business_settings SET value = value + 1 WHERE `key` = 'number_of_message'")
            connection.commit()
    except Exception as e:
        print(f"Error incrementing message count: {e}")

    # Save to memory
    memory.save_context({"user_query": user_query}, {"output": bot_response})

    return JSONResponse(content={
        "reply": bot_response,
        "related_products": [{k: v for k, v in product.items() if k != "marginal_price"} for product in retrieved_products],
        "session_id": session_id
    })


def send_to_facebook(recipient_id: str, message_text: str = None, image_url: str = None):
    """Send message or image back to user via Facebook Graph API."""
    if image_url:
        # Send image
        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": recipient_id},
            "message": {
                "attachment": {
                    "type": "image",
                    "payload": {
                        "url": image_url,
                        "is_reusable": True
                    }
                }
            }
        }
    else:
        # Send text message
        payload = {
            "messaging_type": "RESPONSE",
            "recipient": {"id": recipient_id},
            "message": {"text": message_text}
        }
    
    response = requests.post(
        settings.FB_GRAPH_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        print(f"Error sending message: {response.text}")
    return response.status_code == 200


@router.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    print(f"VERIFY_TOKEN: {settings.VERIFY_TOKEN}, Received token: {token}")  # Debug
    
    if mode == "subscribe" and token == settings.VERIFY_TOKEN:
        return Response(content=str(challenge), status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/webhook")
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
                        "http://127.0.0.1:8000/chat",
                        data={"text": incoming_msg, "session_id": session_id},
                        files=files,
                        timeout=30.0
                    )
                else:
                    response = await client.post(
                        "http://127.0.0.1:8000/chat",
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