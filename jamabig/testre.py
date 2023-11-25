import re

# response_text="Hello! I'm here to help you. Based on the symptoms and imaging findings you provided, I would diagnose the patient with D: Mediastinitis. The patient's persistent fever after esophageal foreign body removal, along with the signs of esophageal perforation and air leak toward the left lobe of the thyroid gland, are consistent with a mediastinitis infection. Please let me know if you have any further questions or if there's anything else I can help you with!"
character_pattern1 = r'option\s*\((\w)\):'

response_text = "Hello! I'm here to help you. Based on the symptoms and imaging findings you provided, I would diagnose the patient with D: Mediastinitis. The patient's persistent fever after esophageal foreign body removal, along with the signs of esophageal perforation and air leak toward the left lobe of the thyroid gland, are consistent with a mediastinitis infection. Please let me know if you have any further questions or if there's anything else I can help you with!"

    # Search for the character pattern in the text
match = re.search(character_pattern1, response_text)

if match:
    # Extract the matched character
    extracted_character = match.group(1)
    print("Extracted Character:", extracted_character)
    dataframe2.at[ind,'Answer_Llama']=extracted_character


if match is None:
    print("yyyyy")
    character_pattern2=r'\(([A-D])\)'
    match = re.search(character_pattern2, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

if match is None:
    print("1111")
    character_pattern3=r'[A-D]:\s*([\w\s-]+)'
    match = re.search(character_pattern3, response_text)

    if match:
        print("yyyyyyyyyy")
        extracted_character = match.group(0)[0]
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character


if match is None:
    character_pattern4=r'([A-D])\. '
    match = re.search(character_pattern4, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

if match is None:
    character_pattern5=r'^([A-D]): '
    match = re.search(character_pattern5, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

if match is None:
    character_pattern6=r'\b([A-D]):\s'
    match = re.search(character_pattern6, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

if match is None:
    character_pattern7=r'[A-D]:\s*([\w\s-]+)'
    match = re.search(character_pattern6, response_text)

    if match:
        extracted_character = match.group(0)[0]
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

if match is None:
    character_pattern8=r'[A-D]:\s*([\w\s-]+)'
    match = re.search(character_pattern6, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)
        dataframe2.at[ind,'Answer_Llama']=extracted_character

else:
    print("Character not found in the text.")