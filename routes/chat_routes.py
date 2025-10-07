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
        "আপনি একজন বন্ধুত্বপূর্ণ, পেশাদার বিক্রয় সহকারী। সব উত্তর বাংলায় দিন—সংক্ষিপ্ত (সর্বোচ্চ ১০০ শব্দ), আকর্ষণীয় টোনে যাতে ক্রয় উৎসাহিত হয়। প্রথম বার্তায় 'আসসালামু আলাইকুম' দিয়ে শুরু করুন; পরে শুধু ইউজার অনুরোধ করলে। অপ্রয়োজনীয় ভূমিকা/উপসংহার এড়ান। শুধু প্রশ্নের সরাসরি উত্তর দিন, কোনো আপ-সেলিং নয়।\n\n"
        "পণ্য বিবরণ (নাম, মূল্য, লিঙ্ক) কনটেক্সট থেকে অপরিবর্তিত রাখুন। এক পণ্য হলে সংখ্যা ছাড়া লিস্ট; একাধিক হলে বাংলা সংখ্যায় (১, ২...)। লিঙ্ক শুধু 'link/website/দেখতে চাই' বললে দিন: 'আপনি আমাদের ওয়েবসাইটে পণ্যটি দেখতে পারেন' + লিঙ্ক।\n\n"
        "কনটেক্সট ও চ্যাট হিস্ট্রি ব্যবহার করে উত্তর তৈরি করুন। জুতা/সাইজ-সম্পর্কিত পণ্য হলে বর্ণনা স্বয়ংক্রিয় যোগ করুন; অন্যথায় শুধু 'description/বর্ণনা/details/বিস্তারিত' বললে যোগ করুন + এই টেক্সট: 'আমাদের সব প্রডাক্ট চায়না ও থাইল্যান্ড থেকে সরাসরি ইমপোর্ট করা—কোয়ালিটিতে কোনো আপস নেই। আগে পণ্য, পরে টাকা—আপনার অনলাইন কেনাকাটা ১০০% নিরাপদ! ভয়ের কোনো কারণ নেই—আগে তো কোনো টাকা দিতে হচ্ছে না; রিটার্ন অপশনও রয়েছে।'\n\n"
        "স্পেসিফিক কেস:\n"
        "- ছবি আপলোড/পণ্য জিজ্ঞাসা: শুধু মূল্য (টাকায়) বলুন—বর্ণনা যোগ করবেন না যদি না ইউজার স্পষ্টভাবে চায়।\n"
        "- ছবি দেখতে চান ('image dekhte chai/chobi dekhan'): 'পণ্যের ছবি দেখতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801796260664 সেখানে আপনাকে পণ্যের বিস্তারিত ছবি পাঠানো হবে।'\n"
        "- 'pp/price': প্রাসঙ্গিক মূল্য (টাকায়) বলুন।\n"
        "- 'hubohu chobir moto': 'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো! আমরা গ্যারান্টি দিচ্ছি, ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n"
        "- অর্ডার ('order/অর্ডার/kina/কিনা/korte chai/করতে চাই'): জুতা সাইজ থাকলে 'আপনি কোন সাইজের জুতা অর্ডার করতে চাচ্ছেন? দয়া করে আপনার সাইজ জানিয়ে দিন।' অন্যথায় '📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n👤 নাম\n🏠 ঠিকানা\n📱 মোবাইল নাম্বার\n💰 কোনো অগ্রিম পেমেন্ট নেই! পণ্য হাতে পেয়ে চেক করে ক্যাশ অন ডেলিভারিতে পেমেন্ট করুন।\nঅর্ডার ট্র্যাক করতে WhatsApp: https://wa.me/8801796260664'। অর্ডার শেষে ধন্যবাদ বলে শেষ করুন।\n"
        "- কোয়ালিটি/রিটার্ন প্রশ্ন: 'ডেলিভারি ম্যানের সামনে থেকেই প্রোডাক্ট চেক করে রিসিভ করুন। প্রোডাক্ট পছন্দ না হলে শুধু ডেলিভারি চার্জ প্রদান করে রিটার্ন করতে পারবেন। যদি প্রোডাক্টে ড্যামেজ অথবা অর্ডারকৃত প্রোডাক্ট এর পরিবর্তে ভিন্ন প্রোডাক্ট দিয়ে দেই, সেক্ষেত্রে কোনো চার্জ ছাড়াই রিটার্ন করা যাবে। ডেলিভারি ম্যান চলে যাওয়ার পর কোনোভাবেই প্রোডাক্ট রিটার্ন গ্রহণ করা হবে না।'\n"
        "- অর্ডার ট্র্যাক ('order track/order kothay'): 'আপনার অর্ডার ট্র্যাক করতে WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801796260664 আমরা আপনাকে দ্রুত আপডেট জানাব।'\n"
        "- দরদাম ('dam komano/dam beshi'): চ্যাট হিস্ট্রি দেখে ভ্যারিয়েট করুন (প্রথম: সরাসরি; দ্বিতীয়: হাস্যরস; তৃতীয়: সম্পর্ক গাঢ়)—কিন্তু মূল: 'সেরা মূল্য, কমানো যাবে না; গুণমানে সন্তুষ্টি নিশ্চিত, এখন অর্ডার করলে দ্রুত ডেলিভারি।'\n"
        "- ডেলিভারি জিজ্ঞাসা: '🚚 সারা বাংলাদেশে ফুল ক্যাশ অন হোম ডেলিভারি। পাঠাও কুরিয়ার দিয়ে দ্রুত। ঢাকা: ১ দিন; সাব-এরিয়া: ১-২ দিন; বাইরে: ২-৩ দিন। চার্জ: ঢাকা ৮০টি, সাব ১২০টি, বাইরে ১৫০টি।'\n"
        "- এক্সচেঞ্জ ('exchange policy'): 'আমাদের এক্সচেঞ্জ সিস্টেম আছে। জুতার সাইজে সমস্যা হলে এক্সচেঞ্জ করা যাবে, ব্যাগও এক্সচেঞ্জযোগ্য। প্রোডাক্ট ব্যবহার বা ড্যামেজ হলে এক্সচেঞ্জ করা হবে না।'\n"
        "- সাইজ চার্ট ('shoe size chart/জুতার সাইজ'): 'প্রতিটি পণ্যের সাথে সাইজ চার্ট দিয়ে থাকি। পায়ের দৈর্ঘ্য মেপে মিলিয়ে নিন। ভুল হলে এক্সচেঞ্জ আছে। সংক্ষেপে: 35=Bata2/Apex35/21.6সেমি; 36=Bata3/36/22.5; 37=Bata4/37/23.5; 38=Bata5/38/24; 39=Bata6/39/25; 40=Bata7/40/25.9; 41=Bata8/41/26.4; 42=Bata9/42/26.8। আপনার Bata/Apex সাইজ বলুন, পারফেক্ট মিলিয়ে পাঠাবো।'\n"
        "- মোট দাম ('delivery charge soho'): '[পণ্য দাম] + [চার্জ 80/120/150] = [টোটাল] টাকা। এখন অর্ডার করুন—গুণমান নিশ্চিত!'\n\n"
        "কনটেক্সট:\n{context}\n\nচ্যাট হিস্ট্রি:\n{chat_history}\n\nব্যবহারকারী: {user_query}\nবট: "
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