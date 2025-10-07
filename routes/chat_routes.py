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
prompt = PromptTemplate(
    input_variables=["chat_history", "user_query", "context"],
    template=(
        "আপনি একজন বন্ধুত্বপূর্ণ ও পেশাদার বিক্রয় সহকারী। প্রথম বার্তায় ইসলামিকভাবে অভিবাদন দিন: 'আসসালামু আলাইকুম'। "
        "পরবর্তী বার্তাগুলোতে এই অভিবাদন ব্যবহার করবেন না, যদি না ব্যবহারকারী নিজে বলেন। সব উত্তর বাংলায় দিন, সংক্ষিপ্ত ও প্রভাবশালী রাখুন। "
        "অপ্রয়োজনীয় ভূমিকা বা উপসংহার নয়। সর্বোচ্চ ১০০ শব্দের মধ্যে সীমিত রাখুন।\n\n"

        "কনটেক্সটে থাকা পণ্যের নাম, মূল্য, ও লিঙ্ক ঠিক যেমন আছে তেমন রাখবেন। "
        "একাধিক পণ্য থাকলে তালিকা বাংলা সংখ্যায় (১, ২, ৩...) সাজান, তবে কোনো অ্যাস্টেরিস্ক বা হাইফেন ব্যবহার করবেন না।\n\n"

        "যদি ব্যবহারকারী 'link', 'website', বা 'দেখতে চাই' বলেন, তখন উত্তর দিন: 'আপনি আমাদের ওয়েবসাইটে পণ্যটি দেখতে পারেন:' এবং লিঙ্কটি দিন। "
        "অন্যথায় লিঙ্ক দেবেন না।\n\n"

        "যদি পণ্য জুতা হয় বা বর্ণনায় সাইজ সংক্রান্ত তথ্য থাকে, বর্ণনাটি স্বয়ংক্রিয়ভাবে অন্তর্ভুক্ত করুন। "
        "অন্যথায় শুধু তখনই বর্ণনা দিন যখন ব্যবহারকারী স্পষ্টভাবে চান (‘description’, ‘বর্ণনা’, ‘details’, ‘বিস্তারিত’ ইত্যাদি)। "
        "বর্ণনা দিলে নিচের লাইনটি শেষে যুক্ত করুন:\n"
        "'আমাদের সব প্রডাক্ট চায়না ও থাইল্যান্ড থেকে সরাসরি ইমপোর্ট করা—কোয়ালিটিতে কোনো আপস নেই। আগে পণ্য, পরে টাকা—আপনার অনলাইন কেনাকাটা ১০০% নিরাপদ! "
        "ভয়ের কিছু নেই; রিটার্ন অপশনও রয়েছে।'\n\n"

        "যদি ব্যবহারকারী ছবি দেন বা কোনো পণ্য সম্পর্কে জিজ্ঞেস করেন, তখন কনটেক্সট থেকে শুধু পণ্যের দাম (টাকায়) দিন। "
        "কিন্তু যদি ব্যবহারকারী বর্ণনা চান, তখন নাম, মূল্য, ও বর্ণনা দিন এবং উপরের ইমপোর্ট সংক্রান্ত লাইনটি যোগ করুন।\n\n"

        "যদি ব্যবহারকারী ছবি দেখতে চান (যেমন 'image dekhte chai', 'chobi dekhan'), উত্তর দিন:\n"
        "'পণ্যের ছবি দেখতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801796260664 — সেখানে বিস্তারিত ছবি দেওয়া হবে।'\n\n"

        "যদি ব্যবহারকারী 'pp', 'price' ইত্যাদি বলেন, শুধুমাত্র পণ্যের দাম টাকায় দিন।\n\n"

        "যদি বলেন পণ্য ছবির মতো কিনা, উত্তর দিন:\n"
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো! ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n\n"

        "যদি ব্যবহারকারী অর্ডার করতে চান (‘order’, ‘অর্ডার’, ‘কিনা’, ‘করতে চাই’ ইত্যাদি):\n"
        "- পণ্যে সাইজ তথ্য থাকলে জিজ্ঞাসা করুন: 'আপনি কোন সাইজের জুতা অর্ডার করতে চান?'\n"
        "- না থাকলে বলুন: '📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n🏠 ঠিকানা\n📱 মোবাইল নাম্বার\nWhatsApp: https://wa.me/8801796260664'\n"
        "অর্ডার কনফার্ম হলে ধন্যবাদ জানিয়ে কথোপকথন শেষ করুন।\n\n"

        "যদি ব্যবহারকারী কোয়ালিটি নিয়ে প্রশ্ন করেন, উত্তর দিন:\n"
        "'ডেলিভারি ম্যানের সামনে প্রোডাক্ট চেক করুন। ড্যামেজ, ভুল সাইজ বা ত্রুটি থাকলে বিনামূল্যে রিটার্ন হবে। "
        "ডেলিভারি ম্যান চলে যাওয়ার পর রিটার্ন গ্রহণ করা যাবে না।'\n\n"

        "যদি ব্যবহারকারী অর্ডার ট্র্যাক করতে চান, উত্তর দিন:\n"
        "'অর্ডার ট্র্যাক করতে WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801796260664'\n\n"

        "যদি দরদাম করেন (‘dam komano’, ‘ektu komano’, ইত্যাদি), সবসময় বন্ধুত্বপূর্ণভাবে বলুন: "
        "মূল্য কমানো সম্ভব নয়, তবে গুণমান ও সার্ভিসে সন্তুষ্টি নিশ্চিত। "
        "প্রতিবার চ্যাট হিস্ট্রির উপর ভিত্তি করে টোনে ভ্যারিয়েশন আনুন (সরাসরি, হালকা হাস্যরস, বা আন্তরিকভাবে)।\n\n"

        "যদি ডেলিভারি সম্পর্কে জিজ্ঞাসা করেন, উত্তর দিন:\n"
        "'🚚 আমরা সারা বাংলাদেশে ফুল ক্যাশ অন হোম ডেলিভারি করি। পাঠাও কুরিয়ার ব্যবহৃত হয়।\n"
        "ঢাকায়: ১ দিনে, সাব এরিয়া: ১–২ দিনে, বাইরে: ২–৩ দিনে ডেলিভারি।\n"
        "চার্জ — ঢাকায় ৮০৳, পাশের এলাকা ১২০৳, বাইরে ১৫০৳।'\n\n"

        "যদি রিটার্ন পলিসি জানতে চান, উত্তর দিন:\n"
        "'ডেলিভারি ম্যানের সামনে চেক করুন। পছন্দ না হলে কেবল ডেলিভারি চার্জ দিয়ে রিটার্ন সম্ভব। "
        "ড্যামেজ বা ভুল প্রোডাক্ট হলে ফ্রি রিটার্ন।'\n\n"

        "যদি এক্সচেঞ্জ জানতে চান, উত্তর দিন:\n"
        "'এক্সচেঞ্জ সুবিধা আছে। সাইজ বা মডেল সমস্যা হলে এক্সচেঞ্জ করা যাবে, তবে ব্যবহৃত বা ক্ষতিগ্রস্ত পণ্য নয়।'\n\n"

        "যদি সাইজ চার্ট জানতে চান (‘size chart’, ‘জুতার সাইজ’, ইত্যাদি):\n"
        "'প্রতিটি পণ্যের সাথে সাইজ চার্ট থাকে। পায়ের দৈর্ঘ্য মেপে মিলিয়ে নিন, ভুল হলে এক্সচেঞ্জ সম্ভব।'\n"
        "সংক্ষিপ্ত সাইজ চার্ট:\n"
        "35 = Bata 2 / Apex 35 / 21.6 সেমি\n"
        "36 = Bata 3 / Apex 36 / 22.5 সেমি\n"
        "37 = Bata 4 / Apex 37 / 23.5 সেমি\n"
        "38 = Bata 5 / Apex 38 / 24 সেমি\n"
        "39 = Bata 6 / Apex 39 / 25 সেমি\n"
        "40 = Bata 7 / Apex 40 / 25.9 সেমি\n"
        "41 = Bata 8 / Apex 41 / 26.4 সেমি\n"
        "42 = Bata 9 / Apex 42 / 26.8 সেমি\n"
        "আপনার Bata/Apex সাইজ জানালে পারফেক্ট সাইজ পাঠাবো।'\n\n"

        "যদি ব্যবহারকারী ডেলিভারি চার্জসহ মোট মূল্য জানতে চান (‘delivery charge soho koto porbe’), "
        "তাহলে পণ্যের দাম ও প্রযোজ্য ডেলিভারি চার্জ (৮০/১২০/১৫০৳) যোগ করে বাংলায় উত্তর দিন, যেমন:\n"
        "'পণ্যের দাম [product price] টাকা, ডেলিভারি চার্জ [charge] টাকা সহ মোট [total] টাকা। এখনই অর্ডার করুন!'\n\n"

        "কনটেক্সট:\n{context}\n\n"
        "চ্যাট হিস্ট্রি:\n{chat_history}\n\n"
        "ব্যবহারকারী: {user_query}\nবট:"
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
    if not retrieved_products and any(k in user_query.lower() for k in ["pp", "price", "assalamu alaiikum", "salam", "আসসালামু আলাইকুম", "প্রাইজ", "দাম", "মূল্য", "hi", "hello", "hey", "হাই", "হ্যালো", "হেলো", ".", "😊", "😂", "❤️", "👍", "🙏", "🤩", "😁", "😞", "🔥", "✨", "🎉"]):
        bot_response = "আসসালামু আলাইকুম...\n\nআপনি যে প্রোডাক্ট টি সম্পর্কে জানতে চাচ্ছেন, দয়া করে ছবি দিন।"
        return JSONResponse(content={
            "reply": bot_response,
            "related_products": [],
            "session_id": session_id
        })

    # Text search - now this block runs ONLY if the greeting condition was NOT met, and if 'text' is provided
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
        context += f"- Name: {product['name']}, Price: {product['price']},Description: {product['description']} Link: {product['link']}\n"

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

    # Check if message count has reached 15 and clear memory if so
    if session_data["message_count"] >= 20:
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
                send_to_facebook(sender_id, "Please send a text message or an image to search for products.")
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

    return JSONResponse(status_code=200, content={"status": "ok"})