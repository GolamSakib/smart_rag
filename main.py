import faiss
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
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

app = FastAPI()

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
    model="mistral:latest",
    base_url="http://localhost:11434/v1",
    # base_url="https://openrouter.ai/api/v1",
    api_key='ollama',
    max_tokens=300,    # Limits output to 300 tokens, as previously specified
    temperature=0.7
    
)

# Updated Prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "You are a helpful sales assistant. ALWAYS respond in English, using a polite, natural, and persuasive conversational tone.\n"
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

@app.post("/chat/")
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
    retrieved_products = session_data["last_products"]  # Start with last retrieved products

    # Image search
    if images:
        retrieved_products = []
        for image_file in images:
            image = Image.open(image_file.file)
            image_embedding = get_image_embedding(image)
            D, I = image_index.search(np.array([image_embedding]).astype('float32'), k=1)
            retrieved_products.append(image_metadata[I[0][0]])
        # Update last_products in session
        session_data["last_products"] = retrieved_products

    # Text search
    if text:
        docs = text_vector_store.similarity_search(text, k=3)
        for doc in docs:
            retrieved_products.append(doc.metadata)
        # Update last_products if new products are found
        session_data["last_products"] = retrieved_products

    # Remove duplicates
    retrieved_products = [dict(t) for t in {tuple(d.items()) for d in retrieved_products}]

    # Build context
    context = "\nAvailable products:\n"
    for product in retrieved_products:
        context += f"- Name: {product['name']}, Description: {product['description']}, Price: {product['price']}, Marginal Price: {product.get('marginal_price', 'N/A')}\n"

    # Define query
    user_query = text.strip() if text else "Provide the name, description, and price of the product in the uploaded image."


    if any(k in user_query.lower() for k in ["order", "buy", "purchase","confirm","অর্ডার", "ক্রয়","কিনতে", "কিনবো", "অর্ডার করতে", "অর্ডার দিন","কনফার্ম"]):
        bot_response = "অনুগ্রহ করে আপনার অর্ডার সম্পূর্ণ করতে নিচের তথ্য দিন:\nআপনার নাম:\nআপনার ঠিকানা:\nআপনার ফোন নাম্বার:"
    elif any(k in user_query.lower() for k in ["delivery", "deliver", "shipping", "ship", "how many days", "when will i get", "how long", "koydin"]):
        bot_response = "আপনি যদি ঢাকায় থাকেন তবে ১ দিনের মধ্যে পণ্য পাবেন, অন্যথায় ২ দিনের মধ্যে।"
    elif any(k in user_query.lower() for k in ["hubohu", "exactly like", "same as picture", "ছবির মত","হুবহু"]):
        bot_response = "হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা নিশ্চিত করি যে আপনি ছবিতে যা দেখছেন, ঠিক তেমনটাই পাবেন।"
    else:
        chat_history = memory.load_memory_variables({})["chat_history"]
        inputs = {"chat_history": chat_history, "user_query": user_query, "context": context}
        response = chain.invoke(inputs)
        bot_response = response.content

    # Save to memory
    memory.save_context({"user_query": user_query}, {"output": bot_response})

    return JSONResponse(content={
        "reply": bot_response,
        "related_products": [{k: v for k, v in product.items() if k != "marginal_price"} for product in retrieved_products],
        "session_id": session_id
    })