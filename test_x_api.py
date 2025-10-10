import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# It is recommended to set the API key as an environment variable for security.
# You can do this by running the following command in your terminal:
# export XAI_API_KEY='your_api_key'
# On Windows, use:
# set XAI_API_KEY='your_api_key'

api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the XAI_API_KEY environment variable.")

url = "https://api.x.ai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "messages": [
        {
            "role": "system",
            "content": "You are a test assistant."
        },
        {
            "role": "user",
            "content": "Testing. Just say hi and hello world and nothing else."
        }
    ],
    "model": "grok-4-fast-reasoning",
    "stream": False,
    "temperature": 0
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for bad status codes

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

