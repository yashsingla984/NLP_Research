import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random
import re

import openai 

max_retries = 3
  
openai.api_key = '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'

dataframe1 = pd.read_excel('output_fin_jama_big2.xlsx')
dataframe2=dataframe1

dataframe2['message_prompt1']=""
dataframe2['message_prompt2']=""
dataframe2['Answer_chatgpt']=""
dataframe2['Explanation']=""
dataframe2['Actual_Correct_option']=""
dataframe2['IsChatGptCorrect']=""
dataframe2['ModifiedMedicalField']=""


for ind in dataframe1.index:
    if ind==1526:
        break
    
    title=dataframe1['Title'][ind]
    caseart=dataframe1['Case'][ind]
    question=dataframe1['MCQ_question'][ind]
    option1=dataframe1['Option1'][ind]
    option2=dataframe1['Option2'][ind]
    option3=dataframe1['Option3'][ind]
    option4=dataframe1['Option4'][ind]
    correct_option=dataframe1['Correct_option'][ind]

    first_prompt="Title: "+title+"\n"+ "Case is: "+caseart
    dataframe2.at[ind,'message_prompt1']=first_prompt
    for attempt in range(max_retries):
        try:
            if len(first_prompt)>=4050:
                print("Word exceed", ind)
                continue

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant."},
                    {"role": "user", "content": first_prompt}
                ]
            )

            # Extract information from the first response
            first_response_text = response.choices[0].message.content
            # Include the information in the second prompt
            second_prompt = f"Prompt 2: Based on the provided information in the first response:\n{first_response_text}\n"+question+" "+"A: "+option1+" ,"+" "+"B: "+option2+" ,"+" "+"C: "+option3+" ,"+" "+"D: "+option4+". "+"Please choose an answer option. The output format is:  (fill in the letter of the answer). Alphabetical letter only"
            # Send the second prompt to GPT-3.5-turbo
            dataframe2.at[ind,'message_prompt2']=second_prompt
            if len(second_prompt)>=4050:
                print("Word exceed", ind)
                continue
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant."},
                    {"role": "user", "content": second_prompt}
                ]
            )

            # Extract the answer generated by the model in response to the second prompt
            correct_option = second_response.choices[0].message.content.strip()
            answer_pattern = re.compile(r'\b([A-D]):', re.DOTALL)
                    #answer_pattern = r'\b([A-D]):'
                    #explanation_pattern = re.compile(r'Explanation:(.*?)$', re.DOTALL)
            answer_match = answer_pattern.search(correct_option)

            # You can use 'correct_option' for further processing or display it as needed
            #print("Correct Option:", answer_match.group(1))

            actual_pattern = re.compile(r'\b([A-D]).', re.DOTALL)
            actual=actual_pattern.search(correct_option)
            if answer_match:
                answer = answer_match.group(1)
                actual_Answer=actual.group(1)
                print(actual_Answer)
                #explanation = explanation_match.group(1).strip()
                dataframe2.at[ind,'Answer_chatgpt']=answer
                dataframe2.at[ind,'Actual_Correct_option']=actual.group(1)

                if answer!=actual_Answer:
                    dataframe2.at[ind,'IsChatGptCorrect']="No"
                else:
                    dataframe2.at[ind,'IsChatGptCorrect']="Yes"
                
                cellvalue=dataframe1.at[ind,'Superclass']
                if pd.isna(cellvalue):
                    print("cellvalue empty")
                    dataframe2.at[ind,'ModifiedMedicalField']=dataframe1['MedicalField'][ind]
                else:
                    dataframe2.at[ind,'ModifiedMedicalField']=dataframe1['Superclass'][ind]
                
                print(ind)
            
            break
        except Exception as e:
            # Handle API error, e.g., print error message
            print(f"API request failed on attempt {attempt + 1}. Error: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    if attempt == max_retries - 1:
        print(f"Max retry attempts reached for question {ind}. Skipping this question.")


dataframe2.to_excel("output_chatgpt_onlyalpha_big_temp_chain2.xlsx")