import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random
from datetime import datetime

dataframe1 = pd.read_excel('output_fin_date.xlsx')

new_df = pd.DataFrame(columns=['URL','Date','Year'])  # Create a new DataFrame
def extract_year(date_str):
    try:
        date_object = datetime.strptime(date_str, "%B %d, %Y")
        return date_object.year
    except ValueError:
        try:
            # Handling cases where the date format is "Month Year"
            date_object = datetime.strptime(date_str, "%B %Y")
            return date_object.year
        except ValueError:
            return None

for ind in dataframe1.index:
    date=dataframe1['Date'][ind]
    url1=dataframe1['URL'][ind]
    #dataframe1['Date'][ind]
    year=extract_year(date)
    print(year)
    new_df=new_df.append({'URL' : url1, 'Date' : date,'Year' : year},ignore_index = True)


new_df.to_excel("year.xlsx")
