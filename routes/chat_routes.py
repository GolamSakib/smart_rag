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
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime,timedelta  
from services.model_manager import model_manager
from config.settings import settings
from services.database_service import db_service






# Keep a small in-memory cache to avoid duplicate processing
processed_messages = set()

router = APIRouter()


# In-memory store for per-session memory
session_memories = defaultdict(lambda: {
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "last_products": [],  # Store last retrieved products
    "message_count": 0    # Track number of messages in the session
})

# Updated Prompt template with discount calculation rule
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "আসসালামু আলাইকুম! আপনি momsandkidsworld এর একজন বন্ধুত্বপূর্ণ, সুপার স্টাইলিশ এবং পেশাদার বিক্রয় সহকারী। আমরা ব্যাগ ও জুতায় রং ছড়াই! প্রথম বার্তায় ইসলামিক অভিবাদন 'আসসালামু আলাইকুম' ব্যবহার করুন, পরবর্তীতে শুধু ব্যবহারকারীর অনুরোধে। উত্তর বাংলায়, সংক্ষিপ্ত, মজাদার এবং উৎসাহী টোনে দিন—ক্রয়ে আগ্রহ জাগাতে! 'প্রিয় গ্রাহক', 'ধন্যবাদ', 'আপনার স্টাইলের জন্য' ব্যবহার করে সম্মান দেখান। গ্রাহকের চাহিদা বুঝে সাজেশন দিন, সুবিধা তুলে ধরুন, আস্থা তৈরি করুন, হালকা হাস্যরস যোগ করে ক্রয়ের দিকে নিয়ে যান—জোর করে নয়।\n"
        
        "যদি ব্যবহারকারী জিজ্ঞাসা করেন আপনি AI কিনা, বলুন: 'আমি momsandkidsworld এর সুপার ফ্রেন্ডলি বিক্রয় সহকারী, আপনার স্টাইল জার্নির সঙ্গী!' \n"
        "উত্তর ৮০ শব্দের মধ্যে রাখুন, মজার টোন বজায় রেখে।\n"
        "অপ্রয়োজনীয় ভূমিকা বা উপসংহার এড়ান।\n"
        "কনটেক্সটের পণ্যের বিবরণ (নাম, মূল্য, লিঙ্ক) অপরিবর্তিত রাখুন। একটি পণ্য হলে সংখ্যা ব্যবহার করবেন না, একাধিক হলে বাংলা সংখ্যা (১, ২, ৩) ব্যবহার করুন।\n"
        "ব্যবহারকারী 'link', 'website' বললে বলুন: 'আপনার স্টাইল দেখতে আমাদের ওয়েবসাইটে চোখ বুলান!' এবং লিঙ্ক দিন।\n"
        "জুতা বা সাইজ-সম্পর্কিত পণ্য হলে বিবরণ স্বয়ংক্রিয়ভাবে দিন। অন্যথায়, 'description', 'বর্ণনা', 'details' চাইলে বিবরণ দিন এবং যোগ করুন: 'আমাদের প্রোডাক্ট চায়না ও থাইল্যান্ড থেকে ইমপোর্ট—কোয়ালিটি টপ-নচ! আগে পণ্য, পরে টাকা—১০০% নিরাপদ কেনাকাটা। রিটার্ন অপশনও আছে, স্টাইল নিয়ে ঝুঁকি নেই!' \n"
        "ছবি আপলোড করলে শুধু মূল্য দিন, বিবরণ চাইলে দিন।\n"
        
        "ছবি দেখতে চাইলে বলুন: 'প্রিয় গ্রাহক, একটু অপেক্ষা করুন, আমাদের মডারেটর এসে আপনার জন্য স্টাইলিশ ছবি দেখাবেন!' \n"
        "'pp', 'price' জিজ্ঞাসা করলে কনটেক্সট থেকে প্রাসঙ্গিক পণ্যের মূল্য দিন।\n"
        "'হুবহু ছবির মতো' জিজ্ঞাসা করলে বলুন: 'একদম ছবির কপি! আমরা গ্যারান্টি দিচ্ছি, আপনার স্টাইল হবে হুবহু ছবির মতো!' \n"
        "অর্ডার করতে চাইলে (জুতার সাইজ থাকলে): 'আপনার পায়ের স্টাইল কোন সাইজে ফিট? দয়া করে সাইজ জানান।' অন্যথায়: '📦 অর্ডার লক করতে দিন:\n🏠 এলাকা (যেমন–চাষাড়া, ধানমন্ডি)\n📱 মোবাইল নাম্বার\n💰 পণ্য হাতে পেয়ে চেক করে ক্যাশ দিন—ঝামেলা ফ্রি!' \n"
        "অর্ডার কনফার্ম হলে প্রমোট করবেন না। ধন্যবাদ দিয়ে শেষ করুন।\n"
        
        "কোয়ালিটি নিয়ে প্রশ্ন হলে: 'ডেলিভারি ম্যানের সামনে প্রোডাক্ট চেক করুন। পছন্দ না হলে শুধু ডেলিভারি চার্জ দিয়ে রিটার্ন করুন। ড্যামেজ বা ভুল পণ্য হলে চার্জ ফ্রি রিটার্ন! ডেলিভারি ম্যান চলে গেলে রিটার্ন গ্রহণ হবে না।' \n"
        "অর্ডার ট্র্যাকিং জিজ্ঞাসা করলে: 'প্রিয় গ্রাহক, একটু ধৈর্য ধরুন! আমাদের মডারেটর সুপার ফাস্ট আপনার অর্ডারের খবর দেবে। কুরিয়ার টিম মেসেজ করবে, অপেক্ষা করুন 💚' \n"
        "অভিযোগ হলে: 'প্রিয় গ্রাহক, টেনশন নেই! আমাদের “Problem Resolve” টিম দ্রুত আপনার সাথে যোগাযোগ করবে। একটু অপেক্ষা করুন, সমাধান পাবেন 💚' \n"
        
        "দরদাম করলে মজার টোনে উত্তর দিন: 'দামটা আমাদের হৃদয়ের মতো—একদম ফিক্সড! কিন্তু গুণমান এমন যে আপনি বিশ্বাস ই করবেন না। এখনই অর্ডার করলে প্রোডাক্ট তাড়াতাড়ি পৌঁছাবে!' (চ্যাট হিস্ট্রি অনুযায়ী ভ্যারিয়েশন আনুন—প্রথমবার সরাসরি, দ্বিতীয়বার মজার টুইস্ট, তৃতীয়বার সম্পর্ক গাঢ় করে।) \n"
        
        "ডেলিভারি জিজ্ঞাসা করলে: '🚚 সারা বাংলাদেশে ক্যাশ অন ডেলিভারি! পাঠাও কুরিয়ারে ঝটপট পৌঁছে যাবে। ঢাকায় ১ দিন, কাছাকাছি এলাকায়(যেমন– কেরানীগঞ্জ, নারায়ণগঞ্জ, সাভার, গাজীপুর) ১-২ দিন, বাইরে ২-৩ দিনে ইনশাআল্লাহ। চার্জ: ঢাকায় ৮০ টাকা, সাব-এলাকায়(নারায়ণগঞ্জ, গাজীপুর, সাভার, কেরানীগঞ্জ) ১৩০ টাকা, বাইরে ১৫০ টাকা। পণ্য হাতে পেয়ে টাকা দিন—ঝক্কি নেই!' \n"
        
        "রিটার্ন পলিসি জিজ্ঞাসা করলে: 'ডেলিভারি ম্যানের সামনে প্রোডাক্ট চেক করুন। পছন্দ না হলে শুধু ডেলিভারি চার্জ দিয়ে রিটার্ন। ড্যামেজ বা ভুল পণ্য হলে ফ্রি রিটার্ন! ডেলিভারি ম্যান চলে গেলে রিটার্ন নেওয়া যাবে না।' \n"
        
        "জুতার সাইজ চার্ট জিজ্ঞাসা করলে: 'আমাদের জুতার সাইজ চার্ট সুপার সিম্পল! Bata/Apex জুতার নিচে সাইজ দেখুন। ৩৫=২১.৬ সেমি, ৩৬=২২.৫ সেমি, ৩৭=২৩.৫ সেমি, ৩৮=২৪ সেমি, ৩৯=২৫ সেমি, ৪০=২৫.৯ সেমি, ৪১=২৬.৪ সেমি, ৪২=২৬.৮ সেমি। আপনার Bata/Apex সাইজ বলুন, পারফেক্ট জুতা পাঠাবো!' \n"
        
        "ডেলিভারি চার্জসহ মোট মূল্য জিজ্ঞাসা করলে: 'পণ্যের দাম [product price] টাকা, ডেলিভারি চার্জ [80/130/150] টাকা সহ মোট [total price] টাকা। এখনই অর্ডার করুন, স্টাইল হাতে পৌঁছে যাবে!' \n"
        
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

def add_to_google_sheet(phone_number: str):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file("google_sheet.json", scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key("1eEuya073QSg0iXsued7e1xJbcrRdKuD7UH7JsyQLvS0").sheet1
        
        today_date = datetime.now().strftime("%Y-%m-%d")
        sheet.insert_row([phone_number, today_date], index=2) # index=2 to insert at the top after header
        print(f"Successfully added {phone_number} to Google Sheet.")
    except Exception as e:
        print(f"Error adding to Google Sheet: {e}")

@router.post("/api/chat")
async def chat(
    images: Optional[List[UploadFile]] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):  
    # bot_response = "Hello"
    # return JSONResponse(content={
    #     "reply": bot_response,
    #     "related_products": [],
    #     "session_id": session_id
    # })
    if not images and not text:
        return JSONResponse(status_code=400, content={"error": "At least one image or text input is required"})
    


    
    session_id = session_id or str(uuid4())
    session_data = session_memories[session_id]
    memory = session_data["memory"]
    retrieved_products = session_data["last_products"]
    session_data["message_count"] += 1  # Increment message count

    # Define query early to allow conditional logic
    user_query = text.strip() if text else "আপলোড করা পণ্যগুলোর নাম এবং মূল্য প্রদান করুন।"

    # Image search - Process images FIRST, before checking for greetings
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

    print("retrieved_products:", retrieved_products)

    # Handle greeting/price query for first-time users with no product context
    # CHECK THIS AFTER image processing but BEFORE text search
    if not retrieved_products and any(k in user_query.lower() for k in ["pp", "price", "assalamu alaiikum", "salam", "আসসালামু আলাইকুম","প্রাইজ","প্রাইস কত","দাম", "মূল্য", "hi", "hello", "hey", "হাই", "হ্যালো", "হেলো", ".", "😊", "😂", "❤️", "👍", "🙏", "🤩", "😁", "😞", "🔥", "✨", "🎉"]):
        bot_response = "আসসালামু আলাইকুম...\n\nআপনি যে প্রোডাক্ট টি সম্পর্কে জানতে চাচ্ছেন, দয়া করে ছবি দিন।"
        return JSONResponse(content={
            "reply": bot_response,
            "related_products": [],
            "session_id": session_id
        })

    # Text search - now this block runs ONLY if the greeting condition was NOT met, and if 'text' is provided
    if text:
        # text_vector_store = model_manager.get_text_vector_store()
        # if text_vector_store is None:
        #     return JSONResponse(status_code=500, content={"error": "Text search not available"})
            
        # docs = text_vector_store.similarity_search(text, k=1)
        # for doc in docs:
        #     retrieved_products.append(doc.metadata)
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
        context += f"- Name: {product['name']}, Price: {product['price']},Description: {product['description']} Link: {product['link']}\n"
    print("Context for LLM:", context)

    # Check for phone number and save to Google Sheet
    phone_pattern = r'(?:\d{8,11}|[০-৯]{8,11})'
    match = re.search(phone_pattern, user_query)
    if match:
        phone_number = match.group(0)
        add_to_google_sheet(phone_number)

    llm = model_manager.get_llm()
    chain = RunnableSequence(prompt | llm)
    chat_history = memory.load_memory_variables({})["chat_history"]
    inputs = {"chat_history": chat_history, "user_query": user_query, "context": context}
    print(inputs)
    response = chain.invoke(inputs)
    bot_response = response.content
    print("Raw bot response:", bot_response)

    # Increment message count in database
    try:
        with db_service.get_cursor() as (cursor, connection):
            cursor.execute("UPDATE business_settings SET value = value + 1 WHERE `key` = 'number_of_message'")
            connection.commit()
    except Exception as e:
        print(f"Error incrementing message count: {e}")

    # Save to memory
    memory.save_context({"user_query": user_query}, {"output": bot_response})

    # Check if message count has reached 3 and clear memory if so
    if session_data["message_count"] >= 30:
        session_memories[session_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "last_products": [],
            "message_count": 0
        }
        print(f"Session memory cleared for session_id: {session_id} after 15 messages")

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

def mark_message_seen(recipient_id: str):
    """Tell Facebook that the bot's messages have been 'seen' by the user."""
    payload = {
        "recipient": {"id": recipient_id},
        "sender_action": "mark_seen"
    }
    response = requests.post(
        settings.FB_GRAPH_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        print(f"Error marking message seen: {response.text}")
        



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
            timestamp_ms = message_data.get("timestamp")
            if timestamp_ms:
                msg_time = datetime.fromtimestamp(timestamp_ms / 1000.0)
                if datetime.utcnow() - msg_time > timedelta(minutes=1):
                    print(f"⚠️ Ignoring old message from {msg_time}")
                    continue

            mid = message_data.get("message", {}).get("mid")
            if mid and mid in processed_messages:
                print(f"Duplicate message ignored: {mid}")
                continue
            if mid:
                processed_messages.add(mid)
    
                if len(processed_messages) > 100:
                    processed_messages.clear()
            sender_id = message_data["sender"]["id"]
            print("Received message_data:", message_data)
            incoming_msg = message_data["message"].get("text", "")
            files = []
            attachments = message_data["message"].get("attachments", [])
            if attachments:
                for idx, attachment in enumerate(attachments):
                    if attachment["type"] == "image":
                        image_url = attachment["payload"]["url"]
                        async with httpx.AsyncClient() as client:
                            image_response = await client.get(image_url)
                            if image_response.status_code == 200:
                                image_content = image_response.content
                                files.append(
                                    (
                                        "images",  
                                        (f"image_{sender_id}_{idx}.jpg", image_content, "image/jpeg")
                                    )
                                )
                            else:
                                print(f" Failed to download image: {image_response.status_code}")
            

            if not incoming_msg and not files:
                send_to_facebook(sender_id, "দয়া করে লিখে বলুন")
                continue

            session_id = sender_id
            async with httpx.AsyncClient() as client:
                print("Sending to /chat:", {"text": incoming_msg, "session_id": session_id, "files_count": len(files)})
                try:
                    if files:
                        response = await client.post(
                            "https://chat.momsandkidsworld.com/api/chat",
                            data={"text": incoming_msg, "session_id": session_id},
                            files=files,
                            timeout=30.0
                        )
                    else:
                        response = await client.post(
                            "https://chat.momsandkidsworld.com/api/chat",
                            data={"text": incoming_msg, "session_id": session_id},
                            timeout=30.0
                        )
                except httpx.RequestError as e:
                    print(f"Network error calling /chat: {e}")
                    send_to_facebook(sender_id, "Sorry, something went wrong while connecting to the chat server.")
                    continue
                if response.status_code != 200:
                    print(f"Error from /chat: {response.status_code}, {response.text}")
                    bot_reply = "Sorry, something went wrong."
                else:
                    result = response.json()
                    bot_reply = result.get("reply", "Sorry, I didn’t understand that.")

            send_to_facebook(sender_id, bot_reply)
            mark_message_seen(sender_id)
            

    return JSONResponse(status_code=200, content={"status": "ok"})