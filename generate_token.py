import requests
import os
from dotenv import load_dotenv

load_dotenv()

# --- STEP 1: Fill in your app details ---
APP_ID = os.getenv("APP_ID")                # from Facebook App Dashboard
APP_SECRET = os.getenv("APP_SECRET")        # from App Settings > Basic
SHORT_LIVED_USER_TOKEN = os.getenv("SHORT_LIVED_USER_TOKEN")  # from Graph API Explorer


exchange_url = (
    f"https://graph.facebook.com/v21.0/oauth/access_token?"
    f"grant_type=fb_exchange_token&"
    f"client_id={APP_ID}&"
    f"client_secret={APP_SECRET}&"
    f"fb_exchange_token={SHORT_LIVED_USER_TOKEN}"
)

print("Exchanging short-lived token for long-lived user token...")
exchange_res = requests.get(exchange_url)
exchange_data = exchange_res.json()

if "access_token" not in exchange_data:
    print("Error exchanging token:", exchange_data)
    exit()

LONG_LIVED_USER_TOKEN = exchange_data["access_token"]
print("Long-lived user token generated!\n")

# --- STEP 3: Get all pages managed by this user and their tokens ---
pages_url = f"https://graph.facebook.com/v21.0/me/accounts?access_token={LONG_LIVED_USER_TOKEN}"
print("Fetching managed pages...")
pages_res = requests.get(pages_url)
pages_data = pages_res.json()

if "data" not in pages_data:
    print("Error fetching pages:", pages_data)
    exit()

for page in pages_data["data"]:
    name = page.get("name")
    page_id = page.get("id")
    page_token = page.get("access_token")

    print(f"   Page: {name}")
    print(f"   Page ID: {page_id}")
    print(f"   Page Access Token: {page_token}\n")

print("Done! Use the Page Access Token above in your Graph API calls.")
