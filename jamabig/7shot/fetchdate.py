import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random

dataframe1 = pd.read_excel('input_jama2.xlsx')

dic={}

#print(dataframe1)
new_df = pd.DataFrame(columns=['URL','Date'])  # Create a new DataFrame
#data
df = pd.DataFrame(columns = ['URL', 'Title'])
for ind in dataframe1.index:
    time.sleep(random.randint(1,10))
    #url = 'http://books.toscrape.com/catalogue/page-' + str(page) + '.html'
    url1=dataframe1['URL'][ind]
    scraper = cloudscraper.create_scraper(delay=10, browser="chrome") 
    content = scraper.get(url1).text
    soup = BeautifulSoup(content, 'html.parser')
    results=soup.findAll("div",{"class":"meta-date"})
    print(ind)
    # if ind==5:
    #     break
    if len(results)==0:
        print("not finded date")
        date_text=""
        new_df=new_df.append({'URL' : url1, 'Date' : date_text},ignore_index = True)
        continue
    #results3=results[0].find_all("p")
    meta_date_div=results[0]
    date_text=""
    if meta_date_div and meta_date_div.find(class_='epreprint'):
        # Find elements with specific classes
        month_element = meta_date_div.find(class_='month')
        day_element = meta_date_div.find(class_='day')
        year_element = meta_date_div.find(class_='year')
        if month_element and day_element and year_element:
            month = month_element.get_text().strip()
            day = day_element.get_text().strip()
            year = year_element.get_text().strip()

            date_text = f"{month} {day} {year}"
            print(date_text)  # Output: October 26, 2023
    else:
        date_text = meta_date_div.get_text().strip()
        print(date_text)  # Output: June 22, 2020

    print(results[0])
    new_df=new_df.append({'URL' : url1, 'Date' : date_text},ignore_index = True)
    #print(new_df)

#     newtextcombined=""
#     for i in results3:
#         #print(i.text)
#         newtextcombined+=i.text+" "
#     print(newtextcombined)
#     title=dataframe1['Value'][ind]
#     dff={'URL' : url1, 'Title' : title, 'Article' : newtextcombined}
#     print(dff)
#     #display(df)
#     df = df.append(dff, ignore_index = True)
#     #print(df.head)
#     #break

print(new_df)
new_df.to_excel("output_fin.xlsx")



