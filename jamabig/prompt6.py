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

dataframe1 = pd.read_excel('output_chatgpt_onlyalpha_big_temp_chain2.xlsx')
dataframe2=dataframe1



for ind in dataframe1.index:
    if ind==1526:
        break
    correct_option=dataframe1['Correct_option'][ind]
    chatgptanswer=dataframe1['Answer_chatgpt'][ind]
    if pd.isna(chatgptanswer):
        print("yash")
        continue

    dataframe2['IsChatGptCorrect'][ind]=""
    actual_pattern = re.compile(r'\b([A-D]).', re.DOTALL)
    actual=actual_pattern.search(correct_option)
    if actual==None:
        dataframe2['Answer_chatgpt'][ind]=""
        dataframe2.at[ind,'Actual_Correct_option']=""
        dataframe2.at[ind,'ModifiedMedicalField']=""
        #ModifiedMedicalField
        continue

    actual_Answer=actual.group(1)
    #explanation = explanation_match.group(1).strip()
    dataframe2.at[ind,'Answer_chatgpt']=chatgptanswer
    dataframe2.at[ind,'Actual_Correct_option']=actual.group(1)

    if chatgptanswer!=actual_Answer:
        #print("bbbbbbbbbbbbbb")
        dataframe2.at[ind,'IsChatGptCorrect']="No"
    else:
        print("bbaa")
        dataframe2.at[ind,'IsChatGptCorrect']="Yes"
    
    print(ind)
            
     


dataframe2.to_excel("output_chatgpt_onlyalpha_big_temp_chain3.xlsx")