import re

# Example string containing the diagnosis
text = "I would choose option A, Cardiac magnetic resonance imaging."

# Define a regular expression pattern to extract the diagnosis option
diagnosis_pattern = r'option\s*([A-D])'  # Pattern to capture the diagnosis option (A, B, C, or D)

# Search for the diagnosis pattern in the text
match = re.search(diagnosis_pattern, text)

if match:
    # Extract the matched diagnosis option
    extracted_diagnosis = match.group(1)  # Extracts the letter (A, B, C, or D)
    print("Extracted Diagnosis Option:", extracted_diagnosis)
else:
    print("Diagnosis option not found in the text.")
