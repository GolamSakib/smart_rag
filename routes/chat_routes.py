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
from datetime import datetime

from services.model_manager import model_manager
from config.settings import settings
from services.database_service import db_service

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
        "আপনি একজন বন্ধুত্বপূর্ণ এবং পেশাদার বিক্রয় সহকারী। প্রতিটি কথোপকথনের প্রথম বার্তায় মুসলিম সাংস্কৃতিক রীতি অনুসারে ইসলামিক অভিবাদন 'আসসালামু আলাইকুম' দিয়ে শুরু করুন। পরবর্তী বার্তাগুলোতে এই অভিবাদন ব্যবহার করবেন না, যদি না ব্যবহারকারী স্পষ্টভাবে এটি অনুরোধ করেন। সব উত্তর বাংলায় হবে, সংক্ষিপ্ত, আকর্ষণীয় এবং বোঝানোর মতো টোন বজায় রাখুন যাতে ক্রয় উৎসাহিত হয়।\n"
        "কনটেক্সটে দেওয়া পণ্যের বিবরণ (নাম, মূল্য, লিঙ্ক) ঠিক যেমন আছে তেমন রাখুন, কোনো অনুবাদ করবেন না। পণ্যের তালিকা প্রদর্শন করার সময় কোনো অ্যাসটেরিস্ক (*) বা হাইফেন (-) ব্যবহার করবেন না। যদি একটি মাত্র পণ্য থাকে, তবে কোনো সংখ্যা (যেমন, ১) ব্যবহার করবেন না, শুধু পণ্যের বিবরণ প্রদর্শন করুন। যদি একাধিক পণ্য থাকে, তবে তালিকাটি বাংলা সংখ্যায় (১, ২, ৩, ইত্যাদি) সাজানো হবে।\n"
        "যদি ব্যবহারকারী সরাসরি লিঙ্ক দেখতে চান বা 'link', 'website', 'দেখতে চাই' এর মতো শব্দ ব্যবহার করেন, তখন তাকে বলুন 'আপনি আমাদের ওয়েবসাইটে পণ্যটি দেখতে পারেন' এবং লিঙ্কটি দিন। অন্যথায় লিঙ্ক দেবেন না।\n"
        "কনটেক্সট এবং চ্যাট হিস্ট্রি ব্যবহার করে ব্যবহারকারীর প্রশ্নের সঠিক এবং আকর্ষণীয় উত্তর দিন।\n"
        "যদি পণ্যটি জুতা হয় অথবা পণ্যের বর্ণনায় সাইজের তথ্য থাকে (যেমন 'size', 'সাইজ', 'জুতা'), তবে পণ্যের বর্ণনা স্বয়ংক্রিয়ভাবে অন্তর্ভুক্ত করুন। অন্যথায়, শুধুমাত্র ব্যবহারকারী স্পষ্টভাবে পণ্যের বর্ণনা চাইলে (যেমন, 'description', 'বর্ণনা', 'details', 'বিস্তারিত' শব্দ ব্যবহার করলে) পণ্যের বর্ণনা অন্তর্ভুক্ত করুন। তখন অবশ্যই নিচের তথ্যটি যোগ করতে হবে:\n"
        "'আমাদের সব প্রডাক্ট চায়না ও থাইল্যান্ড থেকে সরাসরি ইমপোর্ট করা—কোয়ালিটিতে কোনো আপস নেই। আগে পণ্য, পরে টাকা—আপনার অনলাইন কেনাকাটা ১০০% নিরাপদ! ভয়ের কোনো কারণ নেই—আগে তো কোনো টাকা দিতে হচ্ছে না;  রিটার্ন অপশনও রয়েছে'\n"
        "যদি ব্যবহারকারী ছবি আপলোড করেন বা কোনো পণ্য সম্পর্কে জিজ্ঞাসা করেন, তবে পণ্যের নাম এবং মূল্য (টাকায়) অন্তর্ভুক্ত করুন, এবং বর্ণনা শুধুমাত্র তখনই দিন যদি ব্যবহারকারী স্পষ্টভাবে বর্ণনা চান।\n"
        "যদি ব্যবহারকারী পণ্যের ছবি দেখতে চান (যেমন, 'image dekhte chai', 'chobi dekhan', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'পণ্যের ছবি দেখতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801942550295 সেখানে আপনাকে পণ্যের বিস্তারিত ছবি পাঠানো হবে।'\n"
        "যদি ব্যবহারকারী 'pp', 'price', বা অনুরূপ কিছু (কেস-ইনসেনসিটিভ) জিজ্ঞাসা করেন, তবে কনটেক্সট থেকে সবচেয়ে প্রাসঙ্গিক পণ্যের মূল্য শুধুমাত্র টাকায় উল্লেখ করুন।\n"
        "যদি ব্যবহারকারী জিজ্ঞাসা করেন পণ্যটি ছবির মতো কিনা (যেমন, 'hubohu chobir moto'), তবে বাংলায় উত্তর দিন: "
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো! আমরা গ্যারান্টি দিচ্ছি, ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n"
        "যদি ব্যবহারকারী অর্ডার করতে চান, তবে অর্ডার চূড়ান্ত করতে বাংলায় উত্তর দিন: "
        "'📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n"
        "👤 নাম\n"
        "🏠 ঠিকানা\n"
        "📱 মোবাইল নম্বর\n"
        "💰 কোনো অগ্রিম পেমেন্ট নেই! পণ্য হাতে পেয়ে চেক করে ক্যাশ অন ডেলিভারিতে পেমেন্ট করুন।\n"
        "অর্ডার ট্র্যাক করতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801942550295'\n"
        "যদি ব্যবহারকারী অর্ডার ট্র্যাক করতে চান (যেমন, 'order track korte chai', 'order kothay', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'আপনার অর্ডার ট্র্যাক করতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801942550295 আমরা আপনাকে দ্রুত আপডেট জানাব।'\n"
        "যদি ব্যবহারকারী দরদাম করতে চান (যেমন, 'dam komano jay kina', 'ektu komano jay na', 'dam ta onk beshi', বা অনুরূপ), তবে বাংলায় আকর্ষণীয়ভাবে উত্তর দিন: "
        "'আমরা সবসময় সেরা মূল্যে পণ্য বিক্রি করি, এবং এর থেকে কমানো সম্ভব নয়। তবে আমাদের পণ্যের গুণমান ও সেবার নির্ভরযোগ্যতা আপনাকে নিশ্চিতভাবে সন্তুষ্ট করবে! এখনই অর্ডার করলে দ্রুত ডেলিভারি নিশ্চিত।'\n"
        "যদি ব্যবহারকারী ডেলিভারি সম্পর্কে জিজ্ঞাসা করেন, তবে বাংলায় উত্তর দিন: "
        "'🚚 আমরা সারা বাংলাদেশে \"\" ফুল ক্যাশ অন \"\" হোম ডেলিভারি করে থাকি।\n"
        "সহজ ও নিরাপদ ডেলিভারি\n"
        "আপনার বাড়িতেই পণ্য পৌঁছে যাবে, ঝামেলা ছাড়াই।\n"
        "দ্রুত ও নির্ভরযোগ্য ডেলিভারি\n"
        "পাঠাও কুরিয়ারের মাধ্যমে দ্রুত পণ্য পৌঁছানো হয়।\n"
        "👀 পণ্য হাতে দেখে চেক করার সুযোগ\n"
        "পণ্য গ্রহণের সময় ভালো করে পরীক্ষা করে নিতে পারবেন।\n"
        "নিরাপদ পেমেন্ট পদ্ধতি\n"
        "পেমেন্ট শুধুমাত্র পণ্য গ্রহণের পরই দিতে হবে।\n"
        "তারপর\n"
        "ঢাকার মধ্যে: আপনার অর্ডারকৃত পণ্যটি পৌঁছে যাবে ১-২ দিনের মধ্যে।\n"
        "ঢাকার বাইরে: অর্ডারকৃত পণ্যটি ২-৩ দিনের মধ্যে আপনার ঠিকানায় পৌঁছে যাবে ইনশাআল্লাহ।\n"
        "আমরা প্রতিটি অর্ডারে ভালোবাসা ও যত্ন দিয়ে ডেলিভারি নিশ্চিত করি।\n"
        "ডেলিভারি চার্জ:\n"
        "ঢাকার ভিতরে – ৳৮০\n"
        "ঢাকা সংলগ্ন সাব-এলাকা (নারায়ণগঞ্জ, গাজীপুর, সাভার, কেরানীগঞ্জ) – ৳১২০\n"
        "ঢাকার বাইরে – ৳১৫০\n"
        "কোনো প্রকার এডভান্স দিতে হবে না!\n"
        "ডেলিভারির সময় হাতে পেয়েই টাকা দিবেন\n"
        "সেম ডে ডেলিভারি (ঢাকা মেট্রো)\n"
        "অর্ডার সময়: সকাল ৬টা–৯টার মধ্যে অর্ডার করলে একই দিনে ডেলিভারি দেওয়া হবে।\n"
        "ডেলিভারি চার্জ: ৳১২০ (অগ্রিম পরিশোধযোগ্য)।\n"
        "Bkash personal number=01716685128\n"
        "ঢাকার বাইরে সেম ডে ডেলিভারি প্রযোজ্য নয়\n"
        "যদি ব্যবহারকারী রিটার্ন পলিসি সম্পর্কে জিজ্ঞাসা করেন (যেমন, 'return policy', 'ফেরত নীতি', 'exchange policy', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'ডেলিভারি ম্যানের সামনে থেকেই প্রোডাক্ট চেক করে রিসিভ করুন। প্রোডাক্ট পছন্দ না হলে শুধু ডেলিভারি চার্জ প্রদান করে রিটার্ন করতে পারবেন। যদি প্রোডাক্টে ড্যামেজ, ছিঁড়া-ফাটা, চেইন/রঙ উঠে যাওয়া, ঘষা লেগে থাকা বা অর্ডারকৃত জুতার সাইজের পরিবর্তে ভিন্ন জুতার সাইজ আমরা দিয়ে দেই, সেক্ষেত্রে কোনো চার্জ ছাড়াই রিটার্ন করা যাবে। ডেলিভারি ম্যান চলে যাওয়ার পর কোনোভাবেই প্রোডাক্ট রিটার্ন গ্রহণ করা হবে না।'\n"
        "যদি ব্যবহারকারী এক্সচেঞ্জ পলিসি সম্পর্কে জিজ্ঞাসা করেন (যেমন, 'exchange policy', 'এক্সচেঞ্জ নীতি', 'বিনিময় নীতি', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'আমাদের এক্সচেঞ্জ সিস্টেম আছে। জুতার সাইজে সমস্যা হলে এক্সচেঞ্জ করা যাবে, ব্যাগও এক্সচেঞ্জযোগ্য। তবে প্রোডাক্ট রিসিভ করার পর সাবধানে রাখবেন। প্রোডাক্ট ব্যবহার বা ড্যামেজ হলে এক্সচেঞ্জ করা হবে না।'\n"
        "যদি ব্যবহারকারী জুতার সাইজ চার্ট সম্পর্কে জিজ্ঞাসা করেন (যেমন, 'shoe size chart', 'জুতার সাইজ', 'size chart', 'সাইজ চার্ট', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'জুতার সাইজ বোঝার জন্য প্রতিটি প্রোডাক্টের সাথে আমরা সাইজ চার্ট দিয়ে থাকি। আপনি আপনার পায়ের দৈর্ঘ্য ইঞ্চি/সেন্টিমিটারে মেপে চার্টের সাথে মিলিয়ে নিলেই সঠিক সাইজ বেছে নিতে পারবেন। যদি তারপরও মেলাতে সমস্যা হয়, আমাদের সাথে যোগাযোগ করলে আপনাকে গাইড করা হবে। ভুল সাইজ হয়ে গেলে আমাদের exchange সুবিধাও আছে, ইনশাআল্লাহ চিন্তার কিছু নেই।\n\nবাংলাদেশের যেকোনো জনপ্রিয় ব্র্যান্ড যেমন Bata, Apex বা আপনার ব্যবহৃত স্থানীয় ব্র্যান্ডের জুতার নিচে বা ভিতরে সাধারণত সাইজ লেখা থাকে। দয়া করে সেই সাইজটি দেখে আমাদের জানালে আমরা আপনার জন্য একদম পারফেক্ট সাইজের জুতা সাজেস্ট করব। অর৥াৎ, আপনার সাইজ অনুযায়ী জুতাই আপনাকে দেওয়া হবে, ইনশাআল্লাহ।”\n\nআমাদের জুতার সাইজ চার্ট নিচে দেওয়া হলো :\n\nজুতার সাইজ বোঝার নিয়ম সহজ \n\n35 মানে Bata 2 / Apex 35 / পা লম্বা ২১.৬ সেমি\n\n36 মানে Bata 3 / Apex 36 / পা লম্বা ২২.৫ সেমি\n\n37 মানে Bata 4 / Apex 37 / পা লম্বা ২৩.৫ সেমি\n\n38 মানে Bata 5 / Apex 38 / পা লম্বা ২৪ সেমি\n\n39 মানে Bata 6 / Apex 39 / পা লম্বা ২৫ সেমি\n\n40 মানে Bata 7 / Apex 40 / পা লম্বা ২৫.৯ সেমি\n\n41 মানে Bata 8 / Apex 41 / পা লম্বা ২৬.৪ সেমি\n\n42 মানে Bata 9 / Apex 42 / পা লম্বা ২৬.৮ সেমি\n\nআপনি শুধু বলুন, আপনার পায়ের নিচে বা Bata/Apex জুতায় কোন নাম্বার লেখা আছে। সেই অনুযায়ী আমরা সঠিক সাইজের জুতা পাঠাবো।”\n\nআপনার Bata জুতায় যে নাম্বার লেখা আছে সেই নাম্বারটাই বলুন। যেমন Bata 6 মানে আমাদের সাইজ 39, Bata 7 মানে আমাদের সাইজ 40। apex এর সাইজ আর আমাদের জুতার সাইজ সেইম apex এর সাইজ 40 হলে আমাদের জুতার সাইজ ও 40। আপনার Bata/Apex সাইজ জানালেই আমরা একদম পারফেক্ট সাইজের জুতা পাঠাবো।'\n"
        "যদি ব্যবহারকারী ডেলিভারি চার্জসহ মোট মূল্য জানতে চান (যেমন, 'delivery charge soho koto porbe'), তবে পণ্যের তালিকাভুক্ত মূল্যের সাথে ডেলিভারি চার্জ (ঢাকার ভিতরে ৮০ টাকা, ঢাকা সংলগ্ন এলাকা ১২০ টাকা, ঢাকার বাইরে ১৫০ টাকা) যোগ করুন এবং বাংলায় উত্তর দিন, যেমন: "
        "'পণ্যের দাম [product price] টাকা, ডেলিভারি চার্জ [80/120/150] টাকা সহ মোট [total price] টাকা। এখনই অর্ডার করুন!'\n"
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
    if not images and not text:
        return JSONResponse(status_code=400, content={"error": "At least one image or text input is required"})
    
    session_id = session_id or str(uuid4())
    session_data = session_memories[session_id]
    memory = session_data["memory"]
    retrieved_products = session_data["last_products"]
    session_data["message_count"] += 1  # Increment message count

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
        context += f"- Name: {product['name']}, Price: {product['price']},Description: {product['description']} Link: {product['link']}\n"

    # Define query
    user_query = text.strip() if text else "আপলোড করা পণ্যগুলোর নাম এবং মূল্য প্রদান করুন।"

    # Check for phone number and save to Google Sheet
    phone_pattern = r'(?:\d{8,11}|[০-৯]{8,11})'
    match = re.search(phone_pattern, user_query)
    if match:
        phone_number = match.group(0)
        add_to_google_sheet(phone_number)

    # Check if user wants to order and if product has size variants
    if any(k in user_query.lower() for k in ["order", "অর্ডার", "kina", "কিনা", "korte chai", "করতে চাই"]):
        # Check if any product has size information in description
        has_size_variants = False
        size_products = []
        for product in retrieved_products:
            description = product.get('description', '').lower()
            # Check for shoe-specific patterns: "সাইজ=" followed by numbers (shoe sizes)
            # vs bag patterns: "সাইজ=" followed by measurements (cm/inches)
            if 'সাইজ=' in description:
                # Check if it's shoe sizes (numbers like 36,37,38,39,40) vs measurements (cm, inches)
                # Look for shoe size pattern: সাইজ= followed by numbers separated by commas
                shoe_size_pattern = r'সাইজ\s*=\s*সাইজ\s*=\s*\d+(?:,\d+)*'
                # Look for measurement pattern: contains cm, inch, সেমি, ইঞ্চি
                measurement_pattern = r'(সেমি|ইঞ্চি|cm|inch|length|width|height)'
                
                if re.search(shoe_size_pattern, description) and not re.search(measurement_pattern, description):
                    has_size_variants = True
                    size_products.append(product)
                # Also check for explicit shoe keywords as backup
                elif any(shoe_word in description for shoe_word in ['জুতা', 'shoe', 'juta']):
                    has_size_variants = True
                    size_products.append(product)
        
        if has_size_variants:
            bot_response = "আপনি কোন সাইজের জুতা অর্ডার করতে চাচ্ছেন? দয়া করে আপনার সাইজ জানিয়ে দিন।"
        else:
            # Regular order confirmation
            bot_response = "📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n👤 নাম\n🏠 ঠিকানা\n📱 মোবাইল নাম্বার\n💰 কোনো অগ্রিম পেমেন্ট নেই! পণ্য হাতে পেয়ে চেক করে ক্যাশ অন ডেলিভারিতে পেমেন্ট করুন।\nঅর্ডার ট্র্যাক করতে আমাদের WhatsApp-এ যোগাযোগ করুন: https://wa.me/8801942550295"
    elif any(k in user_query.lower() for k in ["hubohu", "exactly like", "same as picture", "ছবির মত", "হুবহু"]):
        bot_response = "হ্যাঁ, পণ্য একদম হুবহু ছবির মতো হবে! আমরা নিশ্চিত করি যে আপনি ছবিতে যা দেখছেন, ঠিক তেমনটাই পাবেন।"
    elif not retrieved_products and any(k in user_query.lower() for k in ["pp", "price", "প্রাইজ", "দাম"]):
        bot_response = "যেই প্রোডাক্ট টির দাম সম্পর্কে জানতে চাচ্ছেন তার ছবি অথবা কোড টি দিন"
    else:
        llm = model_manager.get_llm()
        chain = RunnableSequence(prompt | llm)
        chat_history = memory.load_memory_variables({})["chat_history"]
        inputs = {"chat_history": chat_history, "user_query": user_query, "context": context}
        print(inputs)
        response = chain.invoke(inputs)
        bot_response = response.content

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
    if session_data["message_count"] >= 15:
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
            sender_id = message_data["sender"]["id"]
            print("Received message_data:", message_data)
            incoming_msg = message_data["message"].get("text", "")
            files = []

            # Handle multiple image attachments
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
                                        "images",  # must match parameter name in /api/chat
                                        (f"image_{sender_id}_{idx}.jpg", image_content, "image/jpeg")
                                    )
                                )
                            else:
                                print(f"Failed to download image: {image_response.status_code}")
            
            if not incoming_msg and not files:
                send_to_facebook(sender_id, "Please send a text message or an image to search for products.")
                continue

            session_id = sender_id
            async with httpx.AsyncClient() as client:
                print("Sending to /chat:", {"text": incoming_msg, "session_id": session_id, "files_count": len(files)})
                if files:
                    response = await client.post(
                        "https://chat.momsandkidsworld.com/api/chat",
                        data={"text": incoming_msg, "session_id": session_id},
                        files=files,  # ✅ Now can send multiple images
                        timeout=30.0
                    )
                else:
                    response = await client.post(
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