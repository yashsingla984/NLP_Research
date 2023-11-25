import requests
import json

api_key = "74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01"
url = "https://api.together.ai/v1/chat/70B"

# Set up the headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Set up the payload
payload = {
    "user_id": "123456789",
    "chat_id": "123456789",
    "message": "where is paris ?"
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)
#response_data = response.json()
# Check the response status code
if response.status_code == 200:
    print("Success!")
    print(response["message"])
else:
    print("Error:", response.status_code)
