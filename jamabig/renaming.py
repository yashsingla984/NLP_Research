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

dataframe1 = pd.read_excel('output_palm_onlyalpha_big_temp2.xlsx')



for ind in dataframe1.index:
    if ind==1526:
        break
    orig_name=dataframe1['ModifiedMedicalField'][ind]
    if pd.isna(orig_name):
        continue
    
    #print(orig_name)
    #print(len(orig_name))
    if orig_name=="JAMA Cardiology Clinical Challenge ":
        #print("yashhhh")
        dataframe1.at[ind,'ModifiedMedicalField']="Cardiology"
    
    elif orig_name=="JAMA Cardiology Diagnostic Test Interpretation ":
        dataframe1.at[ind,'ModifiedMedicalField']="Cardiology"

    elif orig_name=="JAMA Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="JAMA"

    elif orig_name=="JAMA Dermatology Clinicopathological Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Dermatology"

    elif orig_name=="JAMA Diagnostic Test Interpretation ":
        dataframe1.at[ind,'ModifiedMedicalField']="Diagnostic"

    elif orig_name=="JAMA Oncology Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Oncology"

    elif orig_name=="JAMA Oncology Diagnostic Test Interpretation ":
        dataframe1.at[ind,'ModifiedMedicalField']="Oncology"

    elif orig_name=="JAMA Ophthalmology Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Ophthalmology"

    elif orig_name=="JAMA Pediatrics Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Pediatrics"

    elif orig_name=="JAMA Psychiatry Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Psychiatry"
        
    elif orig_name=="JAMA Surgery Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Surgery"
    elif orig_name=="JAMA Neurology Clinical Challenge ":
        dataframe1.at[ind,'ModifiedMedicalField']="Neurology"

    elif orig_name=="Pediatric Quality Measures":
        print("yassss")
        dataframe1.at[ind,'ModifiedMedicalField']="Pediactrics"
    


dataframe1.to_excel("output_palm_onlyalpha_big_temp2_renamed.xlsx")
#print("yash")
