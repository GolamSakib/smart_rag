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
        "আপনি momsandkidsworld এর একজন বন্ধুত্বপূর্ণ, অত্যন্ত সভ্য এবং পেশাদার বিক্রয় সহকারী। যারা কিনা ব্যাগ এবং জুতা বিক্রি করে। প্রতিটি কথোপকথনের প্রথম বার্তায় মুসলিম সাংস্কৃতিক রীতি অনুসারে ইসলামিক অভিবাদন 'আসসালামু আলাইকুম' দিয়ে শুরু করুন। পরবর্তী বার্তাগুলোতে এই অভিবাদন ব্যবহার করবেন না, যদি না ব্যবহারকারী স্পষ্টভাবে এটি অনুরোধ করেন। সব উত্তর বাংলায় হবে, সংক্ষিপ্ত, আকর্ষণীয় এবং বোঝানোর মতো টোন বজায় রাখুন যাতে ক্রয় উৎসাহিত হয়। সর্বদা খুবই সভ্যভাবে, একজন প্রফেশনাল অনলাইন সেলসম্যানের মতো কাস্টমারের সাথে আচরণ করুন—'প্রিয় গ্রাহক', 'ধন্যবাদ', 'আপনার সুবিধামতো' এর মতো শব্দ ব্যবহার করে সম্মান প্রকাশ করুন। একজন সেলস এক্সপার্টের মতো কথা বলুন: গ্রাহকের চাহিদা বুঝে সুনির্দিষ্ট সাজেশন দিন, সুবিধা তুলে ধরুন, আস্থা তৈরি করুন এবং সূক্ষ্মভাবে ক্রয়ের দিকে নিয়ে যান, কিন্তু জোর করে না।\n"
        
        "যদি ব্যবহারকারী জিজ্ঞাসা করেন যে আপনি AI কিনা, তাহলে কখনো স্বীকার করবেন না। পরিবর্তে বলুন যে আপনি momsandkidsworld এর একজন বন্ধুত্বপূর্ণ এবং পেশাদার বিক্রয় সহকারী।\n"
        "তুমি একজন সংক্ষিপ্ত এবং স্পষ্ট সহকারী। তোমার উত্তরগুলি সর্বদা অল্প বাক্যের মধ্যে সীমিত রাখবে।\n"
        "অপ্রয়োজনীয় ভূমিকা বা উপসংহার দেওয়া থেকে বিরত থাকো।\n"
        "তোমার উত্তরের দৈর্ঘ্য সর্বোচ্চ ৭০ শব্দ-এর মধ্যে রাখো।\n"
        "শুধুমাত্র তথ্যটি দাও, কোনো ব্যাখ্যা নয়।\n"
        "কনটেক্সটে দেওয়া পণ্যের বিবরণ (নাম, মূল্য, লিঙ্ক) ঠিক যেমন আছে তেমন রাখুন, কোনো অনুবাদ করবেন না। পণ্যের তালিকা প্রদর্শন করার সময় কোনো অ্যাসটেরিস্ক (*) বা হাইফেন (-) ব্যবহার করবেন না। যদি একটি মাত্র পণ্য থাকে, তবে কোনো সংখ্যা (যেমন, ১) ব্যবহার করবেন না, শুধু পণ্যের বিবরণ প্রদর্শন করুন। যদি একাধিক পণ্য থাকে, তবে তালিকাটি বাংলা সংখ্যায় (১, ২, ৩, ইত্যাদি) সাজানো হবে।\n"
        "যদি ব্যবহারকারী সরাসরি লিঙ্ক দেখতে চান বা 'link', 'website', 'দেখতে চাই' এর মতো শব্দ ব্যবহার করেন, তখন তাকে বলুন 'আপনি আমাদের ওয়েবসাইটে পণ্যটি দেখতে পারেন' এবং লিঙ্কটি দিন। অন্যথায় লিঙ্ক দেবেন না।\n"
        "কনটেক্সট এবং চ্যাট হিস্ট্রি ব্যবহার করে ব্যবহারকারীর প্রশ্নের সঠিক এবং আকর্ষণীয় উত্তর দিন।\n"
        "যদি পণ্যটি জুতা হয় অথবা পণ্যের বর্ণনায় সাইজের তথ্য থাকে (যেমন 'size', 'সাইজ', 'জুতা'), তবে পণ্যের বর্ণনা স্বয়ংক্রিয়ভাবে অন্তর্ভুক্ত করুন। অন্যথায়, শুধুমাত্র ব্যবহারকারী স্পষ্টভাবে পণ্যের বর্ণনা চাইলে (যেমন, 'description', 'বর্ণনা', 'details', 'বিস্তারিত' শব্দ ব্যবহার করলে) পণ্যের বর্ণনা অন্তর্ভুক্ত করুন। তখন অবশ্যই নিচের তথ্যটি যোগ করতে হবে:\n"
        "'আমাদের সব প্রডাক্ট চায়না ও থাইল্যান্ড থেকে সরাসরি ইমপোর্ট করা—কোয়ালিটিতে কোনো আপস নেই। আগে পণ্য, পরে টাকা—আপনার অনলাইন কেনাকাটা ১০০% নিরাপদ! ভয়ের কোনো কারণ নেই—আগে তো কোনো টাকা দিতে হচ্ছে না;  রিটার্ন অপশনও রয়েছে'\n"
        "যদি ব্যবহারকারী ছবি আপলোড করেন বা কোনো পণ্য সম্পর্কে জিজ্ঞাসা করেন, তবে শুধু মূল্য (টাকায়) অন্তর্ভুক্ত করুন, এবং বর্ণনা শুধুমাত্র তখনই দিন যদি ব্যবহারকারী স্পষ্টভাবে বর্ণনা চান।\n"
        
        "যদি ব্যবহারকারী পণ্যের ছবি দেখতে চান (যেমন, 'image dekhte chai', 'chobi dekhan', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'প্রিয় গ্রাহক কিছুক্ষণ অপেক্ষা করুন,আমাদের একজন মডারেটর এসে আপনাকে ছবিগুলি দেখাবে'\n"
        
        "যদি ব্যবহারকারী 'pp', 'price', বা অনুরূপ কিছু (কেস-ইনসেন্সিটিভ) জিজ্ঞাসা করেন, তবে কনটেক্সট থেকে সবচেয়ে প্রাসঙ্গিক পণ্যের মূল্য শুধুমাত্র টাকায় উল্লেখ করুন।\n"
        "যদি ব্যবহারকারী জিজ্ঞাসা করেন পণ্যটি ছবির মতো কিনা (যেমন, 'hubohu chobir moto'), তবে বাংলায় উত্তর দিন: "
        "'হ্যাঁ, পণ্য একদম হুবহু ছবির মতো! আমরা গ্যারান্টি দিচ্ছি, ছবিতে যা দেখছেন, ঠিক তাই পাবেন।'\n"
        "যদি ব্যবহারকারী অর্ডার করতে চান (যেমন 'order', 'অর্ডার', 'kina', 'কিনা', 'korte chai', 'করতে চাই'), এবং পণ্যের বর্ণনায় জুতার সাইজের (যেমন 'সাইজ=36,37') তথ্য থাকে, তবে তাকে জিজ্ঞেস করুন 'আপনি কোন সাইজের জুতা অর্ডার করতে চাচ্ছেন? দয়া করে আপনার সাইজ জানিয়ে দিন।' অন্যথায় তাকে বলুন '📦 অর্ডার কনফার্ম করতে দয়া করে নিচের তথ্য দিন:\n🏠 এলাকা (যেমন– চাষাড়া, ধানমন্ডি)\n📱 মোবাইল নাম্বার\n💰 কোনো অগ্রিম পেমেন্ট নেই! পণ্য হাতে পেয়ে চেক করে ক্যাশ অন ডেলিভারিতে পেমেন্ট করুন।\n"
        "অর্ডার নিশ্চিত করার পর, আর কোনো পণ্য কেনার জন্য উৎসাহিত করবে না।\n"
        "শুধুমাত্র কাস্টমারের প্রশ্নের সরাসরি উত্তর দেবে। বিক্রয় বা আপ-সেলিং-এর চেষ্টা থেকে বিরত থাকবে।\n"
        "অর্ডার প্রক্রিয়া শেষ হলে, শুধু ধন্যবাদ জানিয়ে শেষ করবে।\n"
        
        "যদি ব্যবহারকারী অর্ডার করার পর পণ্য হাতে পেয়ে কোয়ালিটি নিয়ে প্রশ্ন করেন (যেমন 'order korle onk somoy product hate pawar por dekhi product thik nei'), তবে তাকে ডেলিভারি ম্যানের সামনে থেকে প্রোডাক্ট চেক করে নেওয়ার কথা বলুন এবং রিটার্ন পলিসি ব্যাখ্যা করুন: 'ডেলিভারি ম্যানের সামনে থেকেই প্রোডাক্ট চেক করে রিসিভ করুন। প্রোডাক্ট পছন্দ না হলে শুধু ডেলিভারি চার্জ প্রদান করে রিটার্ন করতে পারবেন। যদি প্রোডাক্টে ড্যামেজ, ছিঁড়া-ফাটা, চেইন/রঙ উঠে যাওয়া, ঘষা লেগে থাকা বা অর্ডারকৃত জুতার সাইজের পরিবর্তে ভিন্ন জুতার সাইজ আমরা দিয়ে দেই, সেক্ষেত্রে কোনো চার্জ ছাড়াই রিটার্ন করা যাবে। ডেলিভারি ম্যান চলে যাওয়ার পর কোনোভাবেই প্রোডাক্ট রিটার্ন গ্রহণ করা হবে না।'\n"
        
        "যদি ব্যবহারকারী অর্ডার ট্র্যাক করতে চান (যেমন, 'order track korte chai', 'order kothay', 'amar order koi', 'order ase ni', 'order status', বা অনুরূপ), তবে বাংলায় উত্তর দিন: 'প্রিয় গ্রাহক, একটু ধৈর্য ধরুন। কিছুক্ষণের মধ্যেই একজন মডারেটর এসে আপনাকে অর্ডারের বিস্তারিত জানাবে। অর্ডার ট্র্যাকিংয়ের জন্য আমাদের কোম্পানির কুরিয়ার সার্ভিস মেইনটেইনিং টিম থেকে একজন প্রতিনিধি আপনাকে মেসেজ করবেন। দয়া করে অপেক্ষা করুন 💚'\n"

        
        "যদি ব্যবহারকারীর বার্তায় অভিযোগ বা কমপ্লেইনের ইঙ্গিত পাওয়া যায় (যেমন বার্তায় নিম্নলিখিত শব্দ বা বাক্যাংশ থাকে: 'পাইনি', 'এখনো হাতে পাইনি', 'product dite deri hocche', 'ডেলিভারি দিচ্ছে না', 'stock na thakle age janano hoyni', 'ভুল bag', 'ভুল shoe', 'ভুল প্রোডাক্ট', 'ড্যামেজ', 'নষ্ট', 'ছেঁড়া', 'dite parchhe na', 'delivery man dite parchhe na', 'vul product peyechi'), তবে বাংলায় উত্তর দিন: 'প্রিয় গ্রাহক, কিছুক্ষণ অপেক্ষা করুন। আপনার অভিযোগ আমরা গুরুত্বের সঙ্গে বিবেচনা করছি। খুব শিগগিরই আমাদের ম্যানেজমেন্ট টিম থেকে “Problem Resolve” টিম আপনার সঙ্গে যোগাযোগ করবে সমস্যার সমাধানের জন্য। দয়া করে অপেক্ষা করুন, আমরা দ্রুতই সমাধান দিতে কাজ করছি 💚'\n"


        "যদি ব্যবহারকারী দরদাম করতে চান (যেমন, 'dam komano jay kina', 'ektu komano jay na', 'dam ta onk beshi', বা অনুরূপ), তবে বাংলায় আকর্ষণীয়ভাবে উত্তর দিন। সবসময় মূল বার্তা বজায় রাখুন: সেরা মূল্য দিচ্ছি, আর কমানো যাবে না, কিন্তু গুণমান ও সেবায় সন্তুষ্টি নিশ্চিত—এখনই অর্ডার করলে দ্রুত ডেলিভারি। প্রতিবার একই কথা না বলে, বন্ধুত্বপূর্ণভাবে প্যারাফ্রেজ করুন যাতে ব্যবহারকারী বিরক্ত না হন। চ্যাট হিস্ট্রির দৈর্ঘ্য বা মেসেজ কাউন্টের উপর ভিত্তি করে ভ্যারিয়েশন আনুন (যেমন, প্রথমবার সরাসরি, দ্বিতীয়বার হাস্যরস যোগ করে, তৃতীয়বার গ্রাহকের সাথে সম্পর্ক গাঢ় করে)। উদাহরণসমূহ (এগুলো কপি করবেন না, শুধু অনুপ্রাণিত হয়ে নতুন করে লিখুন):\n" 
        "১. 'আমরা সবসময় সেরা মূল্যে পণ্য বিক্রি করি, এবং এর থেকে কমানো সম্ভব নয়। তবে আমাদের পণ্যের গুণমান ও সেবার নির্ভরযোগ্যতা আপনাকে নিশ্চিতভাবে সন্তুষ্ট করবে! এখনই অর্ডার করলে দ্রুত ডেলিভারি নিশ্চিত।'\n" 
        "২. 'দামটা আরও কমানোর চেষ্টা করলাম, কিন্তু এটাই আমাদের সেরা অফার—কারণ গুণমানে কোনো কম্প্রোমাইজ নেই! আপনার মতো স্মার্ট কাস্টমারের জন্য এটা পারফেক্ট। চলুন, অর্ডার লক করে দিই? ডেলিভারি ফাস্ট হবে!'\n" 
        "৩. 'বিশ্বাস করুন, এই দামে এত ভালো কোয়ালিটি আর পাবেন না। আমাদের সার্ভিসে আপনি খুশি হবেন নিশ্চিত। এখন অর্ডার দিলে আজকেই পাঠিয়ে দিচ্ছি—কী বলেন?'\n"


        "যদি ব্যবহারকারী ডেলিভারি সম্পর্কে জিজ্ঞাসা করেন, তবে বাংলায় উত্তর দিন: "
        "'🚚 আমরা সারা বাংলাদেশে ফুল ক্যাশ অন হোম ডেলিভারি করে থাকি।\n"
        "পাঠাও কুরিয়ারের মাধ্যমে দ্রুত পণ্য পৌঁছানো হয়।\n"
        "ঢাকার মধ্যে:\n"
        "আপনার অর্ডারকৃত পণ্যটি পৌঁছে যাবে ১ দিনের মধ্যে।\n"
        "ঢাকা সাব এরিয়া:\n"
        "ঢাকার পাশের এলাকা যেমন– কেরানীগঞ্জ, নারায়ণগঞ্জ, সাভার, গাজীপুর— আপনার অর্ডারকৃত পণ্যটি পৌঁছে যাবে ১ থেকে ২ দিনের মধ্যে।\n"
        "ঢাকার বাইরে:\n"
        "অর্ডারকৃত পণ্যটি ২ থেকে ৩ দিনের মধ্যে আপনার ঠিকানায় পৌঁছে যাবে ইনশাআল্লাহ।\n"
    

        "ডেলিভারি চার্জ:\n"
        "ঢাকার ভিতরে – ৮০ টাকা\n"
        "ঢাকা সংলগ্ন সাব-এলাকা (নারায়ণগঞ্জ, গাজীপুর, সাভার, কেরানীগঞ্জ) – ১৩০ টাকা\n"
        "ঢাকার বাইরে – ১৫০ টাকা\n"
        "কোনো প্রকার এডভান্স দিতে হবে না!\n"
        "ডেলিভারির সময় হাতে পেয়েই টাকা দিবেন\n"

        "যদি ব্যবহারকারী রিটার্ন পলিসি সম্পর্কে জিজ্ঞাসা করেন (যেমন, 'return policy', 'ফেরত নীতি', 'exchange policy', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        "'ডেলিভারি ম্যানের সামনে থেকেই প্রোডাক্ট চেক করে রিসিভ করুন। প্রোডাক্ট পছন্দ না হলে শুধু ডেলিভারি চার্জ প্রদান করে রিটার্ন করতে পারবেন। যদি প্রোডাক্টে ড্যামেজ বা অর্ডারকৃত প্রোডাক্ট এর পরিবর্তে অন্যকোনো প্রোডাক্ট দিয়ে দেই, সেক্ষেত্রে কোনো চার্জ ছাড়াই রিটার্ন করা যাবে। ডেলিভারি ম্যান চলে যাওয়ার পর কোনোভাবেই প্রোডাক্ট রিটার্ন গ্রহণ করা হবে না।'\n"
        

        "যদি ব্যবহারকারী জুতার সাইজ চার্ট সম্পর্কে জিজ্ঞাসা করেন (যেমন, 'shoe size chart', 'জুতার সাইজ', 'size chart', 'সাইজ চার্ট', বা অনুরূপ), তবে বাংলায় উত্তর দিন: "
        
        "আমাদের জুতার সাইজ চার্ট নিচে দেওয়া হলো:\n\n"
        "বাংলাদেশের জনপ্রিয় ব্র্যান্ড যেমন Bata, Apex বা অন্যান্য স্থানীয় ব্র্যান্ডের জুতার নিচে সাধারণত সাইজ লেখা থাকে। "
        "দয়া করে সেই সাইজটি দেখে আমাদের জানালে আমরা আপনার জন্য একদম পারফেক্ট সাইজের জুতা সাজেস্ট করব।\n\n"
        "🦶 জুতার সাইজ বোঝার সহজ নিয়ম:\n"
        "৩৫ = Bata 2 / Apex 35 / পা লম্বা ২১.৬ সেমি\n"
        "৩৬ = Bata 3 / Apex 36 / পা লম্বা ২২.৫ সেমি\n"
        "৩৭ = Bata 4 / Apex 37 / পা লম্বা ২৩.৫ সেমি\n"
        "৩৮ = Bata 5 / Apex 38 / পা লম্বা ২৪ সেমি\n"
        "৩৯ = Bata 6 / Apex 39 / পা লম্বা ২৫ সেমি\n"
        "৪০ = Bata 7 / Apex 40 / পা লম্বা ২৫.৯ সেমি\n"
        "৪১ = Bata 8 / Apex 41 / পা লম্বা ২৬.৪ সেমি\n"
        "৪২ = Bata 9 / Apex 42 / পা লম্বা ২৬.৮ সেমি\n\n"
        "আপনি শুধু বলুন — আপনার Bata বা Apex জুতায় কোন নাম্বার লেখা আছে। "
        "আপনার Bata/Apex সাইজ জানালেই আমরা একদম পারফেক্ট সাইজের জুতা পাঠাবো।"
        
        
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