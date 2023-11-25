import together
import pandas as pd
import json
from tqdm import tqdm
import re
import time
together.api_key = "74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01"

max_retries = 3
model_name = "togethercomputer/llama-2-7b-chat"

#questions = df["Question"].tolist()

responses_dict = {}
dataframe1 = pd.read_excel('output_fin_jama_big2.xlsx')
dataframe2=dataframe1

dataframe2['message_prompt1']=""
dataframe2['Answer_LLama']=""
dataframe2['Explanation']=""
dataframe2["Actual_correct_option"]=""
dataframe2['IsLLamaCorrect']=""
dataframe2['ModifiedMedicalField']=""


for ind in dataframe1.index:
    if ind==1526:
        break
    
    print(ind)
    title=dataframe1['Title'][ind]
    caseart=dataframe1['Case'][ind]
    question=dataframe1['MCQ_question'][ind]
    option1=dataframe1['Option1'][ind]
    option2=dataframe1['Option2'][ind]
    option3=dataframe1['Option3'][ind]
    option4=dataframe1['Option4'][ind]
    correct_option=dataframe1['Correct_option'][ind]

    message="Title: "+title+". Case is: "+caseart+question+" "+"A: "+option1+" ,"+" "+"B: "+option2+" ,"+" "+"C: "+option3+" ,"+" "+"D: "+option4+". "+"Please choose an answer option and I don't require any explaination. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
    dataframe2.at[ind,'message_prompt1']=message
    question=message




#question="Title: A Young Man With a Neck Mass and Hypereosinophilia Case is: A 31-year-old man presented with left cervical and left inguinal masses. He reported intermittent itching and night sweats for 2 years. He denied fever, weight loss, shortness of breath, rashes, diarrhea, and neurological symptoms. On a preemployment evaluation, the patient was told he had a high white blood cell count 2 years ago. On examination, there was left cervical and inguinal lymphadenopathy and no other organomegaly. Complete blood cell count and peripheral blood smear showed marked leukocytosis, with a white blood cell count of 22 340/μL, an absolute neutrophil count of 5360/μL, and 55% eosinophils with an absolute eosinophil count of 12 290/μL (to convert all to cells ×109/L, multiply by 0.001). Vitamin B12 was markedly elevated at more than 4000 pg/mL (to convert to pmol/L, multiply by 0.7378). The erythrocyte sedimentation rate was 5 mm/h. Lactate dehydrogenase was 180 U/L, and alkaline phosphatase was 81 U/L (to convert both to μkat/L, multiply by 0.0167). Evaluations for HIV and hepatitis B and C were all negative. Serum creatinine was 0.76 mg/dL (to convert to μmol/L, multiply by 88.4); alanine aminotransferase and aspartate aminotransferase were 11 U/L and 9.9 U/L, respectively (to convert to μkat/L, multiply by 0.0167); and total bilirubin was 0.35 mg/dL (to convert to μmol/L, multiply by 17.104). Bone marrow biopsy showed hypercellular marrow (cellularity of 100%), myeloid hyperplasia, increased eosinophils with some dysplasia, and a blast count of 2%. Positron emission tomographic–computed tomographic scan showed a left upper cervical lymph node of 2.6 cm and a left inguinal lymph node of 3.1 × 2.3 cm with an standardized uptake value max of 5.7 (Figure, A). Left inguinal lymph node biopsy showed partial involvement by atypical cells with high proliferation index (Ki-67 >95%) that were positive for CD3, CD4, CD8, BCL2, and TDT, suggestive of T-cell lymphoblastic lymphoma/leukemia (Figure, B).A, Positron emission tomographic–computed tomographic (PET/CT) scan of the head and neck at presentation showing a left upper cervical lymph node of 2.6 cm (arrowhead). B, Lymph node biopsy immunohistochemical stain with terminal deoxynucleotidyl transferase. The inset shows interphase fluorescence in situ hybridization for FIP1L1::PDGFRA rearrangement (positive). C, PET/CT 12 weeks after treatment initiation.Myeloid/lymphoid neoplasms with eosinophilia and tyrosine kinase gene fusionsWhat Is Your Diagnosis? A: Kimura disease , B: Classic Hodgkin lymphoma , C: T-cell acute lymphoblastic lymphoma/leukemia , D: Myeloid/lymphoid neoplasms with eosinophilia and tyrosine kinase gene fusions. Please choose an answer option and I don't require any explaination. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
#for question in tqdm(questions):
    for attempt in range(max_retries):
        try:
            output = together.Complete.create(
                prompt = f"<human>: {question}\n<bot>:",
                model = model_name,
                max_tokens = 512,
                temperature = 0.8,
                top_k = 60,
                top_p = 0.6,
                repetition_penalty = 1.1,
            stop = ['<human>', '\n\n\n\n']
            )

        # Store generated response in the dictionary with question as the key
            response_text = output['output']['choices'][0]['text'].strip()
            dataframe2.at[ind,'Explanation']=response_text
            print(response_text)


            selected_option_pattern = r'[A-D]:\s*(.*)'




            character_pattern1 = r'option\s*\((\w)\):'

            # Search for the character pattern in the text
            match = re.search(character_pattern1, response_text)

            if match:
                # Extract the matched character
                extracted_character = match.group(1)
                print("Extracted Character:", extracted_character)
                dataframe2.at[ind,'Answer_Llama']=extracted_character


            if match is None:
                character_pattern2=r'\(([A-D])\)'
                match = re.search(character_pattern2, response_text)

                if match:
                    extracted_character = match.group(1)
                    print("Extracted Character:", extracted_character)
                    dataframe2.at[ind,'Answer_Llama']=extracted_character

            if match is None:
                character_pattern3=r'[A-D]:\s*([\w\s-]+)'
                match = re.search(character_pattern3, response_text)

                if match:
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
                match = re.search(character_pattern7, response_text)

                if match:
                    extracted_character = match.group(0)[0]
                    print("Extracted Character:", extracted_character)
                    dataframe2.at[ind,'Answer_Llama']=extracted_character

            if match is None:
                character_pattern8=r'[A-D]:\s*([\w\s-]+)'
                match = re.search(character_pattern8, response_text)

                if match:
                    extracted_character = match.group(1)
                    print("Extracted Character:", extracted_character)
                    dataframe2.at[ind,'Answer_Llama']=extracted_character


            if match is None:
                character_pattern9=r'[A-D](?=\s*-\s*[\w\s-]+)'
                match = re.search(character_pattern9, response_text)

                if match:
                    extracted_character = match.group(1)
                    print("Extracted Character:", extracted_character)
                    dataframe2.at[ind,'Answer_Llama']=extracted_character

            if match is None:
                character_pattern10=r'letter\s*\"([A-D])\"'
                match = re.search(character_pattern10, response_text)

                if match:
                    extracted_character = match.group(1)
                    print("Extracted Character:", extracted_character)
                    dataframe2.at[ind,'Answer_Llama']=extracted_character

            # if match is None:
            #     character_pattern11=r'option\s*([A-D])'
            #     match = re.search(character_pattern11, response_text)

            #     if match:
            #         extracted_character = match.group(1)
            #         print("Extracted Character:", extracted_character)
            #         dataframe2.at[ind,'Answer_Llama']=extracted_character

            elif match is None:
                print("Character not found in the text.")
            

            
            if match:
                answer=extracted_character
                actual_pattern = re.compile(r'\b([A-D]).', re.DOTALL)
                actual=actual_pattern.search(correct_option)
                actual_Answer=actual.group(1)
                dataframe2.at[ind,"Actual_correct_option"]=actual_Answer
                if answer!=actual_Answer:
                    dataframe2.at[ind,'IsLLamaCorrect']="No"
                else:
                    dataframe2.at[ind,'IsLLamaCorrect']="Yes"
                        
                cellvalue=dataframe1.at[ind,'Superclass']
                if pd.isna(cellvalue):
                    print("cellvalue empty")
                    dataframe2.at[ind,'ModifiedMedicalField']=dataframe1['MedicalField'][ind]
                else:
                    dataframe2.at[ind,'ModifiedMedicalField']=dataframe1['Superclass'][ind]
            break
        except Exception as e:
            # Handle API error, e.g., print error message
            print(f"API request failed on attempt {attempt + 1}. Error: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff


dataframe2.to_excel("output_llama_onlyalpha_temp_big1.xlsx")
print("yasssh")
