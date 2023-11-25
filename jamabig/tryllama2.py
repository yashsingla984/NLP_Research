import together
import pandas as pd
import json
from tqdm import tqdm
import re
together.api_key = "74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01"


model_name = "togethercomputer/llama-2-7b-chat"

#questions = df["Question"].tolist()


responses_dict = {}
question="Title: A Young Man With a Neck Mass and Hypereosinophilia Case is: A 31-year-old man presented with left cervical and left inguinal masses. He reported intermittent itching and night sweats for 2 years. He denied fever, weight loss, shortness of breath, rashes, diarrhea, and neurological symptoms. On a preemployment evaluation, the patient was told he had a high white blood cell count 2 years ago. On examination, there was left cervical and inguinal lymphadenopathy and no other organomegaly. Complete blood cell count and peripheral blood smear showed marked leukocytosis, with a white blood cell count of 22 340/μL, an absolute neutrophil count of 5360/μL, and 55% eosinophils with an absolute eosinophil count of 12 290/μL (to convert all to cells ×109/L, multiply by 0.001). Vitamin B12 was markedly elevated at more than 4000 pg/mL (to convert to pmol/L, multiply by 0.7378). The erythrocyte sedimentation rate was 5 mm/h. Lactate dehydrogenase was 180 U/L, and alkaline phosphatase was 81 U/L (to convert both to μkat/L, multiply by 0.0167). Evaluations for HIV and hepatitis B and C were all negative. Serum creatinine was 0.76 mg/dL (to convert to μmol/L, multiply by 88.4); alanine aminotransferase and aspartate aminotransferase were 11 U/L and 9.9 U/L, respectively (to convert to μkat/L, multiply by 0.0167); and total bilirubin was 0.35 mg/dL (to convert to μmol/L, multiply by 17.104). Bone marrow biopsy showed hypercellular marrow (cellularity of 100%), myeloid hyperplasia, increased eosinophils with some dysplasia, and a blast count of 2%. Positron emission tomographic–computed tomographic scan showed a left upper cervical lymph node of 2.6 cm and a left inguinal lymph node of 3.1 × 2.3 cm with an standardized uptake value max of 5.7 (Figure, A). Left inguinal lymph node biopsy showed partial involvement by atypical cells with high proliferation index (Ki-67 >95%) that were positive for CD3, CD4, CD8, BCL2, and TDT, suggestive of T-cell lymphoblastic lymphoma/leukemia (Figure, B).A, Positron emission tomographic–computed tomographic (PET/CT) scan of the head and neck at presentation showing a left upper cervical lymph node of 2.6 cm (arrowhead). B, Lymph node biopsy immunohistochemical stain with terminal deoxynucleotidyl transferase. The inset shows interphase fluorescence in situ hybridization for FIP1L1::PDGFRA rearrangement (positive). C, PET/CT 12 weeks after treatment initiation.Myeloid/lymphoid neoplasms with eosinophilia and tyrosine kinase gene fusionsWhat Is Your Diagnosis? A: Kimura disease , B: Classic Hodgkin lymphoma , C: T-cell acute lymphoblastic lymphoma/leukemia , D: Myeloid/lymphoid neoplasms with eosinophilia and tyrosine kinase gene fusions. Please choose an answer option and I don't require any explaination. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
#for question in tqdm(questions):
output = together.Complete.create(
    prompt = f"<human>: {question}\n<bot>:",
    model = model_name,
    max_tokens = 512,
    temperature = 0.8,
    top_k = 60,
    top_p = 0.6,
    repetition_penalty = 1.1,
   stop = ['<human>', '\n\n\n']
)

# Store generated response in the dictionary with question as the key
response_text = output['output']['choices'][0]['text'].strip()
#responses_dict[question] = response_text
print(response_text)


#question="Title: Tzanck Smear of Ulcerated Plaques. Case is: A man in his 30s with AIDS presented with acute-onset painful scattered umbilicated papulopustules and ovoid ulcerated plaques with elevated, pink borders on the face, trunk, and extremities (Figure, A). The patient also had a new-onset cough but was afebrile and denied other systemic symptoms. Due to his significant immunocompromise, the clinical presentation was highly suspicious for infection. For rapid bedside differentiation of multiple infectious etiologies, a Tzanck smear was performed by scraping the base of an ulcerated lesion and inner aspect of a pseudopustule and scraping its base with a #15 blade. These contents were placed on a glass slide, fixed, and stained with Wright-Giemsa and subsequently Papanicolaou staining to further characterize the changes seen.A, Clinical image demonstrating papulopustules and ovoid ulcerated plaques with elevated, pink borders on the elbows. B, Tzanck smear using Wright-Giemsa staining of specimen demonstrating ballooning of keratinocytes and peripheralization of nuclear material (original magnification ×20).What Is Your Diagnosis? A: Herpes simplex virus , B: Histoplasmosis , C: Molluscum contagiosum , D: Mpox. Please choose an answer option and I don't require any explaination. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
#question="Title: Peripheral Blasts in a Patient Receiving Chemotherapy Case is: An 80-year-old man with stage II bladder carcinoma (T2NXM0) and atrial fibrillation treated with apixaban presented to the emergency department with 1 week of fatigue and 2 days of dyspnea on exertion. One week prior to presentation, he received a fourth cycle of carboplatin/gemcitabine for bladder carcinoma with 6 mg of pegylated granulocyte colony-stimulating factor (G-CSF). The patient reported no anorexia, fever, melena, hematemesis, hematuria, cough, orthopnea, or peripheral edema.His vital signs were normal except for a heart rate of 103/min. His white blood cell count was 22 × 103/μL (reference, 4-11 × 103/μL), increased from 4.8 × 103/μL 8 days prior. His manual differential, which was previously normal, showed 18% bands (0%-10%), 2% metamyelocytes, 7% myelocytes, 7% promyelocytes, and 6% blasts. His hemoglobin level was 5.2 g/dL (reference, 13-17 g/dL), decreased from 7.4 g/dL, and platelets were 25 × 103/μL (reference, 150-420 × 103/μL), decreased from 268 × 103/μL 8 days prior. Ferritin was 1423 ng/mL (reference, 300-400 ng/mL). Mean corpuscular volume, prothrombin time, international normalized ratio, partial thromboplastin time, fibrinogen, haptoglobin, vitamin B12, and methylmalonic acid values were normal, and results of a direct antiglobulin test were negative. A computed tomography (CT) scan of his abdomen and pelvis was normal. He received 2 units of packed red blood cells and was admitted to the hospital. Flow cytometry identified a small population of CD34+/CD117+ cells (Figure).Left, Peripheral blood smear showing normocytic anemia with anisopoikilocytosis and leukocytosis with 6% to 8% blast forms. Right, Flow cytometry of peripheral blood demonstrating a small population of white blood cells that stained positive for CD34 and CD117, which are markers of immature myeloblasts.Esophagogastroduodenoscopy revealed 2 nonbleeding angioectasias in the stomach that were treated with argon plasma coagulation. Three days after admission, his white blood cell count was 27.7 × 103/μL with 4% peripheral blasts, hemoglobin was 7.3 g/dL, and platelet count had increased to 92 × 103/μL without a platelet transfusion.Repeat complete blood cell count with differential in 1 to 2 weeksWhat Would You Do Next? A: Perform a bone marrow biopsy , B: Prescribe all-trans retinoic acid , C: Repeat complete blood cell count with differential in 1 to 2 weeks , D: Start cytoreductive therapy with hydroxyurea. Please choose an answer option and I don't require any explaination. The output format is:  (fill in the letter of the answer). Alphabetical letter only"

# diagnosis_pattern = r'\([A-Z]\)\s+(\w+)'

# # Search for the diagnosis pattern in the text
# diagnosis_match = re.search(diagnosis_pattern, response_text)

# if diagnosis_match:
#     # Extract the diagnosis found in the text
#     extracted_diagnosis = diagnosis_match.group(0)
#     print("Extracted Diagnosis:", extracted_diagnosis)
# else:
#     print("Diagnosis not found in the text.")

selected_option_pattern = r'[A-D]:\s*(.*)'

# Search for the selected option pattern in the text
# selected_option_match = re.search(selected_option_pattern, response_text)

# if selected_option_match:
#     # Extract the selected answer option
#     selected_option = selected_option_match.group(0)
#     print("Selected Answer Option:", selected_option)
# else:
#     print("Selected answer option not found in the text.")



# answer_option_pattern = r'[A-D]\W*[:\)]?\s*([\w\s]+)'

# # Function to extract answer options from text
# def extract_answer_option(text):
#     matches = re.findall(answer_option_pattern, text)
#     if matches:
#         return matches[0].strip()

# # Extract answer option from text
# extracted_option = extract_answer_option(response_text)
# #extracted_option2 = extract_answer_option(text2)

# # Display extracted answer options
# print("Extracted Option 1:", extracted_option)

character_pattern1 = r'option\s*\((\w)\):'

# Search for the character pattern in the text
match = re.search(character_pattern1, response_text)

if match:
    # Extract the matched character
    extracted_character = match.group(1)
    print("Extracted Character:", extracted_character)

elif match is None:
    character_pattern2=r'\(([A-D])\)'
    match = re.search(character_pattern2, response_text)

    if match:
        extracted_character = match.group(1)
        print("Extracted Character:", extracted_character)


else:
    print("Character not found in the text.")






#print("Extracted Option 2:", extracted_option2)
# # Save responses to a JSON file
# with open("responses.json", "w") as json_file:
#     json.dump(responses_dict, json_file, indent=4)

# print("Responses saved to 'responses.json'") 