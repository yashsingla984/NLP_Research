# import requests

# def ask_question(api_key, question):
#     url = "https://api.llama2.com/ask"
#     headers = {"Authorization": f"Bearer {api_key}"}
#     data = {"question": question}

#     response = requests.post(url, headers=headers, json=data)

#     if response.status_code == 200:
#         result = response.json()
#         return result
#     else:
#         print(f"Error: {response.status_code}, {response.text}")
#         return None

# # Replace 'YOUR_API_KEY' with your actual API key
# api_key = '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'
# question = "What is the capital of France?"

# result = ask_question(api_key, question)

# if result:
#     print("Response:", result['answer'])



# import replicate
# output = replicate.run(
#     "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
#     input={"prompt": ...}
# )
# # The meta/llama-2-70b-chat model can stream output as it's running.
# # The predict method returns an iterator, and you can iterate over that output.
# for item in output:
#     print(item)



# import requests
# api_key = "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"
# question = "What is the meaning of life?"
# response = requests.get(f"https://api.llama.com/v1/questions?q={question}&api_key={api_key}")
# if response.status_code == 200:
#     print(response.json()["answer"])
# else:
#     print("Failed to get answer")


import requests

# Replace 'YOUR_API_KEY' and the API endpoint with the actual values from Together AI's documentation
# api_key = '74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01'
# endpoint = 'https://api.togetherai.co/v1/endpoint'

# headers = {
#     'Authorization': f'Bearer {api_key}',
#     'Content-Type': 'application/json'  # Adjust content type as per API requirements
# }

# # Make a GET request
# response = requests.get(endpoint, headers=headers)

# # Handle response
# if response.status_code == 200:
#     data = response.json()
#     # Process the data here
# else:
#     print(f"Request failed with status code {response.status_code}")
#     print(response.text)  # Print error message or details



import requests

url = "https://api.together.xyz/inference"

api_key = '74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01'

# payload = {
#     "model": "togethercomputer/RedPajama-INCITE-7B-Instruct",
#     "prompt": "The capital of France is",
#     "max_tokens": 128,
#     "stop": ".",
#     "temperature": 0.7,
#     "top_p": 0.7,
#     "top_k": 50,
#     "repetition_penalty": 1
# }
# headers = {
#     "accept": "application/json",
#     "content-type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.text)


# from langchain.llms import Together

# llm = Together(
#     model="togethercomputer/llama-2-70b",
#     temperature=0.7,
#     max_tokens=1024,
#     top_k=1,
#     together_api_key=api_key
# )

# input_ = """Where is Paris ?"""
# print(llm(input_))
import json

import requests

url = "https://api.together.xyz/inference"

payload = {
    "model": "togethercomputer/llama-2-70b-chat",
    "prompt": "Title: Tzanck Smear of Ulcerated Plaques Case is: A man in his 30s with AIDS presented with acute-onset painful scattered umbilicated papulopustules and ovoid ulcerated plaques with elevated, pink borders on the face, trunk, and extremities (Figure, A). The patient also had a new-onset cough but was afebrile and denied other systemic symptoms. Due to his significant immunocompromise, the clinical presentation was highly suspicious for infection. For rapid bedside differentiation of multiple infectious etiologies, a Tzanck smear was performed by scraping the base of an ulcerated lesion and inner aspect of a pseudopustule and scraping its base with a #15 blade. These contents were placed on a glass slide, fixed, and stained with Wright-Giemsa and subsequently Papanicolaou staining to further characterize the changes seen.A, Clinical image demonstrating papulopustules and ovoid ulcerated plaques with elevated, pink borders on the elbows. B, Tzanck smear using Wright-Giemsa staining of specimen demonstrating ballooning of keratinocytes and peripheralization of nuclear material (original magnification ×20).What Is Your Diagnosis? A: Herpes simplex virus , B: Histoplasmosis , C: Molluscum contagiosum , D: Mpox. Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only",
    "max_tokens": 128,
    "stop": ".",
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer 610328ca940764472a6293fa6cc150a8aec51df08cfd1052a40364f083def75a"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
parsed_response = json.loads(response.text)

# Extract the output part from the JSON
output = parsed_response.get('output', {}).get('choices', [{}])[0].get('text', '')

# Display the output or result
print(output)
