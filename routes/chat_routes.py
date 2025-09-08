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
import re

from services.model_manager import model_manager
from config.settings import settings
from services.database_service import db_service

router = APIRouter()

# In-memory store for per-session memory
session_memories = defaultdict(lambda: {
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "last_products": []  # Store last retrieved products
})

# Updated Prompt template with discount calculation rule
prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "আপনি একজন বন্ধুত্বপূর্ণ এবং পেশাদার বিক্রয় সহকারী। প্রতিটি কথোপকথনের প্রথম বার্তায় মুসলিম সাংস্কৃতিক রীতি অনুসারে ইসলামিক অভিবাদন 'আসসালামু আলাইকুম' দিয়ে শুরু করুন। পরবর্তী বার্তাগুলোতে এই অভিবাদন ব্যবহার করবেন না, যদি না ব্যবহারকারী স্পষ্টভাবে এটি অনুরোধ করেন। সব উত্তর বাংলায় হবে, সংক্ষিপ্ত, আকর্ষণীয় এবং বোঝানোর মতো টোন বজায় রাখুন যাতে ক্রয় উৎসাহিত হয়।\n"
        "কনটেক্সটে দেওয়া পণ্যের বিবরণ (নাম, বর্ণনা, মূল্য) ঠিক যেমন আছে তেমন রাখুন, কোনো অনুবাদ করবেন না।\n"
        "কনটেক্সট এবং চ্যাট হিস্ট্রি ব্যবহার করে ব্যবহারকারীর প্রশ্নের সঠিক এবং আকর্ষণীয় উত্তর দিন।\n"
        "যদি ব্যবহারকারী ছবি আপলোড করেন বা কোনো পণ্য সম্পর্কে জিজ্ঞাসা করেন, তবে পণ্যের নাম এবং মূল্য (টাকায়) অন্তর্ভুক্ত করুন। ব্যবহারকারী স্পষ্টভাবে বর্ণনা চান না বলে উল্লেখ করলে বর্ণনা বাদ দিন।\n"
        "যদি ব্যবহারকারী পণ্যের ছবি দেখতে চান (যেমন, 'image dekhte chai', 'chobi dekhan', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'পণ্যের ছবি দেখতে আমাদের WhatsApp নাম্বারে যোগাযোগ করুন: 01942550295 সেখানে আপনাকে পণ্যের বিস্তারিত ছবি পাঠানো হবে।'\n"
        "যদি ব্যবহারকারী 'pp', 'price', বা অনুরূপ কিছু (কেস-ইনসেনসিটিভ) জিজ্ঞাসা করেন, তবে কনটেক্সট থেকে সবচেয়ে প্রাসঙ্গিক পণ্যের মূল্য শুধুমাত্র টাকায় উল্লেখ করুন।\n"
        "যদি ব্যবহারকারী জিজ্ঞাসা করেন পণ্যটি ছবির মতো কিনা (যেমন, 'hubohu chobir moto'), তবে বাংলায় উত্তর দিন: "
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো! আমরা গ্যারান্টি দিচ্ছি, ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n"
        "যদি ব্যবহারকারী অর্ডার করতে চান, তবে অর্ডার চূড়ান্ত করতে বাংলায় উত্তর দিন: "
        "'📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n"
        "👤 নাম\n"
        "🏠 ঠিকানা\n"
        "📱 মোবাইল নাম্বার\n"
        "💰 কোনো অগ্রিম পেমেন্ট নেই! পণ্য হাতে পেয়ে চেক করে ক্যাশ অন ডেলিভারিতে পেমেন্ট করুন।\n"
        "অর্ডার ট্র্যাক করতে আমাদের WhatsApp নাম্বারে যোগাযোগ করুন: 01942550295'\n"
        "যদি ব্যবহারকারী অর্ডার ট্র্যাক করতে চান (যেমন, 'order track korte chai', 'order kothay', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'আপনার অর্ডার ট্র্যাক করতে আমাদের WhatsApp নাম্বারে যোগাযোগ করুন: 01942550295 আমরা আপনাকে দ্রুত আপডেট জানাব।'\n"
        "যদি ব্যবহারকারী দরদাম করতে চান (যেমন, 'dam komano jay kina', 'ektu komano jay na', 'dam ta onk beshi', বা অনুরূপ), তবে বাংলায় আকর্ষণীয়ভাবে উত্তর দিন: "
        "'আমরা সবসময় সেরা মূল্যে পণ্য বিক্রি করি, এবং এর থেকে কমানো সম্ভব নয়। তবে আমাদের পণ্যের গুণমান ও সেবার নির্ভরযোগ্যতা আপনাকে নিশ্চিতভাবে সন্তুষ্ট করবে! এখনই অর্ডার করলে দ্রুত ডেলিভারি নিশ্চিত।'\n"
        "যদি ব্যবহারকারী ডেলিভারি সম্পর্কে জিজ্ঞাসা করেন, তবে বাংলায় উত্তর দিন: "
        "'🚚 আমরা সারা বাংলাদেশে ক্যাশ অন ডেলিভারি সুবিধা দিচ্ছি:\n"
        "🏠 ঢাকার মধ্যে: ডেলিভারি চার্জ ৮০ টাকা, ১-২ দিনের মধ্যে পৌঁছে যাবে।\n"
        "🚚 ঢাকার বাইরে: ডেলিভারি চার্জ ১২০ টাকা, ২-৩ দিনের মধ্যে পৌঁছে যাবে, ইনশাআল্লাহ।\n"
        "👀 পণ্য হাতে পেয়ে চেক করুন, তারপর পেমেন্ট করুন।\n"
        "🎁 প্রতিটি অর্ডারে দ্রুত ও নির্ভরযোগ্য ডেলিভারি নিশ্চিত।'\n"
        "যদি ব্যবহারকারী ডেলিভারি চার্জসহ মোট মূল্য জানতে চান (যেমন, 'delivery charge soho koto porbe'), তবে পণ্যের তালিকাভুক্ত মূল্যের সাথে ডেলিভারি চার্জ (ঢাকায় ৮০ টাকা, ঢাকার বাইরে ১২০ টাকা) যোগ করুন এবং বাংলায় উত্তর দিন, যেমন: "
        "'পণ্যের দাম [product price] টাকা, ডেলিভারি চার্জ [80/120] টাকা সহ মোট [total price] টাকা। এখনই অর্ডার করুন!'\n"
        "পণ্যের গুণমান, নির্ভরযোগ্যতা এবং জরুরি ভিত্তিতে অর্ডারের আকর্ষণ বাড়ান।\n\n"
        "কনটেক্সট:\n{context}\n\n"
        "চ্যাট হিস্ট্রি:\n{chat_history}\n\n"
        "ব্যবহারকারী: {user_query}\nবট: "
    )
)


def validate_offer_price(response: str, products: List[dict]) -> str:
    """
    Validate the offered price in the bot's response to ensure it is not below the marginal price.
    If below, adjust to the marginal price of the most relevant product.
    """
    if not products:
        return response
    
    # Assume the first product is the most relevant
    marginal_price = float(products[0]["marginal_price"])
    print(f"Marginal price of the most relevant product: {marginal_price}")
    
    # Extract the offered price from the response (assuming format like "[number] টাকা")
    match = re.search(r'(\d+\.?\d*)\s*টাকা', response)
    if match:
        offered_price = float(match.group(1))
        if offered_price < marginal_price:
            # Replace the offered price with the marginal price
            response = re.sub(r'\d+\.?\d*\s*টাকা', f"{int(marginal_price)} টাকা", response)
    
    return response

@router.post("/api/chat")
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
            
        docs = text_vector_store.similarity_search(text, k=1)
        for doc in docs:
            retrieved_products.append(doc.metadata)
        session_data["last_products"] = retrieved_products

    # Remove duplicates
    seen_products = set()
    unique_products = []
    for product in retrieved_products:
        identifier = (product.get('name', '').strip(), product.get('code', '').strip())
        if identifier not in seen_products:
            seen_products.add(identifier)
            unique_products.append(product)
    retrieved_products = unique_products

    # Build context
    context = "\nAvailable products:\n"
    for product in retrieved_products:
        context += f"- Name: {product['name']}, Description: {product['description']}, Price: {product['price']}\n"
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
        
    #     bargaining_keywords = [
    #     "dam komano", "ektu komano", "dam ta onk", "eto dam kno", "komano jay kina", "komano jay na",
    #     "dam kombe", "kom koren", "kom kore den", "dam onik beshi", "onek dami", "koto discount",
    #     "discount pabo", "sera dam", "offer ache", "kom korun", "dam beshi", "kom dame", "discount din",
    #     "price reduce", "bargain", "too expensive", "lower price", "can you reduce", "dam koman",
    #     "dam ta kom korun", "ektu kom korun", "dam onek beshi", "kom daben", "discount diben", "beshi dam",
    #     "দাম কমানো", "একটু কমানো", "দামটা অনেক", "এত দাম কেনো", "কমানো যায় কিনা", "কমানো যায় না",
    #     "দাম কমবে", "কম করেন", "কম করে দেন", "দাম অনেক বেশি", "অনেক দামি", "কত ডিসকাউন্ট",
    #     "ডিসকাউন্ট পাবো", "সেরা দাম", "অফার আছে", "কম করুন", "দাম বেশি", "কম দামে", "ডিসকাউন্ট দিন",
    #     "দাম কমান", "দামটা কম করুন", "একটু কম করুন", "দাম অনেক বেশি", "কম দাবেন", "ডিসকাউন্ট দিবেন",
    #     "বেশি দাম"
    # ]

    # if any(k in user_query.lower() for k in bargaining_keywords):
    #   bot_response = validate_offer_price(bot_response, retrieved_products)

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
    
    print(f"VERIFY_TOKEN: {settings.VERIFY_TOKEN}, Received token: {token}")
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
            print("Received message_data:", message_data)
            incoming_msg = message_data["message"].get("text", "")
            files = []

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

            if not incoming_msg and not files:
                send_to_facebook(sender_id, "Please send a text message or an image to search for products.")
                continue

            session_id = sender_id
            async with httpx.AsyncClient() as client:
                print("Sending to /chat:", {"text": incoming_msg, "session_id": session_id, "files": bool(files)})
                if files:
                    response = await client.post(
                        # "http://127.0.0.1:8000/api/chat",
                        "https://chat.momsandkidsworld.com/api/chat",
                        data={"text": incoming_msg, "session_id": session_id},
                        files=files,
                        timeout=30.0
                    )
                else:
                    response = await client.post(
                        # "http://127.0.0.1:8000/api/chat",
                        "https://chat.momsandkidsworld.com/api/chat",
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