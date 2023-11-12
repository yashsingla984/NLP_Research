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
  
import google.generativeai as palm

# Set your API key
palm.configure(api_key="AIzaSyDdw-SShnw5zyGh7F51GTqM4qP8nCD2Gps")
model_id="models/text-bison-001"



dataframe1 = pd.read_excel('output_fin_jama_big2.xlsx')
dataframe2=dataframe1

dataframe2['message']=""
dataframe2['Answer_chatgpt']=""
dataframe2['Explanation']=""
dataframe2['Actual_Correct_option']=""
dataframe2['IsPalmCorrect']=""
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
    

    #prompt message for chatgpt
    #message="Title: "+title+" Case is: "+caseart+question+" "+"A: "+option1+" ,"+" "+"B: "+option2+" ,"+" "+"C: "+option3+" ,"+" "+"D: "+option4+". "+"Please choose an answer option and explain why that might be correct while the rest are incorrect. The output format is: Answer: (fill in the letter of the answer) Explanation:"
    message="Title: "+title+" Case is: "+caseart+question+" "+"A: "+option1+" ,"+" "+"B: "+option2+" ,"+" "+"C: "+option3+" ,"+" "+"D: "+option4+". "+"Please choose an answer option. The output format is:  (fill in the letter of the answer)"
    #print(message)
    if len(message)>=4050:
        print("Word exceed", ind)
        continue
    
    prompt=message
    for attempt in range(max_retries):
        try:
            completion=palm.generate_text(model=model_id,prompt=prompt,temperature=0.99,max_output_tokens=800,)
            answer=completion.result
            if pd.isna(answer):
                print("Answer not available")
                continue
            print(answer)
            actual_pattern = re.compile(r'\b([A-D]).', re.DOTALL)
            actual=actual_pattern.search(correct_option)
            actual_Answer=actual.group(1)
            #print(actual)
            dataframe2.at[ind,'Answer_Palm']=answer
            dataframe2.at[ind,'Actual_Correct_option']=actual.group(1)

            if answer!=actual_Answer:
                dataframe2.at[ind,'IsPalmCorrect']="No"
            else:
                dataframe2.at[ind,'IsPalmCorrect']="Yes"
                
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
            print("api request failed")
            #print(f"API request failed on attempt {attempt + 1}. Error: {str(e)}")
            time.sleep(2 ** max_retries)  # Exponential backoff
    if attempt == max_retries - 1:
        print(f"Max retry attempts reached for question {ind}. Skipping this question.")   
        



#print(dataframe2.iloc[0])
dataframe2.to_excel("output_palm_onlyalpha_big_temp2.xlsx")
# print(dataframe2.iloc[0])
print("yassssh")
#print(messages)


    