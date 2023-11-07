import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random
import subprocess

dataframe1 = pd.read_excel('outputjama_big1.xlsx')

dic={}

import sys

# Construct the file path relative to the current script's directory
file_path = "./train.txt"

# Open the file for appending
file=open(file_path, "a")
    # Redirect sys.stdout to the file
sys.stdout = file


# df = pd.DataFrame(columns = ['URL', 'Title','Case','Discussion','MCQ_question','Option1','Option2','Option3','Option4','Diagnosis','Correct_option','HasImage','MedicalField'])
df = pd.DataFrame(columns = ['URL', 'Title','Case','Discussion','MCQ_question','Option1','Option2','Option3','Option4','Diagnosis','Correct_option','HasImage','MedicalField','Superclass'])
# def extract_paragraphs(tempsoup):
#     paragraphs = []
#     diagnosis=""
#     fetchtrue=False
#     founddiagnosis=False
#     fetchoptiontrue=False
#     chooseoption=""
#     if tempsoup:
#         article_para=tempsoup.find('div', class_='article-full-text')
#         for paragraph in article_para.find_all('p'):
#             if paragraph.get_text()=="Article Information":
#                 break
#             if fetchoptiontrue==True:
#                 chooseoption=paragraph.get_text()
#                 fetchoptiontrue=False
#                 #print("I have choosen option: ",chooseoption)
#                 #print(chooseoption)
         
#             elif fetchtrue==True:
#                 diagnosis=paragraph.get_text()
#                 fetchtrue=False
#                 founddiagnosis=True
#                 #print("The diagnosis is: ",diagnosis)
#                 #print(diagnosis)
#             elif paragraph.get_text()=="Diagnosis":
#                 fetchtrue=True
#             elif paragraph.get_text()=="What to Do Next":
#                 fetchoptiontrue=True
#                 founddiagnosis=False
        
#             else:
#                 if np.char.count(paragraph.get_text(), ' ') + 1 <8:
#                     continue
#                 paragraphs.append(paragraph.get_text())
#         if founddiagnosis==True and fetchoptiontrue==False:
#             chooseoption=diagnosis
#     return paragraphs,diagnosis,chooseoption


def extract_paragraphs(tempsoup):
    casepara=[]
    case=0
    discusiion=0
    discussionpara=[]
    paragraphs = []
    diagnosis=""
    fetchtrue=False
    founddiagnosis=False
    fetchoptiontrue=False
    chooseoption=""
    inclpara=0
    #print("in extract para")
    if tempsoup:
        #print("in temp soup")
        article_para=tempsoup.find('div', class_='article-full-text')
        stop_condition="Article Information"
        for paragraph in article_para.find_all(['div', 'p']):
            if paragraph.name == 'div':
                if stop_condition in paragraph.get_text():
                    break
            
            else:
                #print(element.get_text())
                if fetchoptiontrue==True:
                    chooseoption=paragraph.get_text()
                    fetchoptiontrue=False
                
                elif fetchtrue==True:
                    diagnosis=paragraph.get_text()
                    fetchtrue=False
                    founddiagnosis=True

                elif paragraph.get_text()=="Diagnosis":
                    fetchtrue=True
                # or paragraph.get_text()== "What To Do Next"
                elif paragraph.get_text()=="What to Do Next" or paragraph.get_text()== "What To Do Next" or paragraph.get_text()== "Answer":
                    if paragraph.get_text()=="Answer" or paragraph.get_text()== "What To Do Next":
                        print(paragraph.get_text())
                    fetchoptiontrue=True
                    founddiagnosis=False
                else:
                    if paragraph.get_text()=="Case":
                        case=1
                    if paragraph.get_text()=="Discussion":
                        case=0
                        discusiion=1

                    if np.char.count(paragraph.get_text(), ' ') + 1 <8:
                        continue
                    
                    if case==1:
                        casepara.append(paragraph.get_text())
                    
                    if discusiion==1:
                        discussionpara.append(paragraph.get_text())
                    paragraphs.append(paragraph.get_text())

        if founddiagnosis==True and fetchoptiontrue==False:
            chooseoption=diagnosis
    return paragraphs,diagnosis,chooseoption,casepara,discussionpara


def hasImage(tempsoup):
    article_para = tempsoup.find('div', class_='article-full-text')

    if article_para:
        # Check if there is an image in the article-full-text div
        image_div = article_para.find('div', class_='figure-table-image')
        if image_div and image_div.find('img'):
            #print("The 'article-full-text' div contains an image.")
            return True 
    return False


# def tellfield(tempsoup):
#     article_para = tempsoup.find('div', class_='meta-article-type thm-col')
#     #print("Journal: ",article_para.get_text())

#     return article_para.get_text()

def tellfield(tempsoup):
    article_para = tempsoup.find('div', class_='meta-article-type thm-col')
    super_class = tempsoup.find('div', class_='meta-super-class')
    if super_class:
        return article_para.get_text(),super_class.get_text()
    #print("Journal: ",article_para.get_text())
    return article_para.get_text(),None
    

def extractMCQ(tempsoup):
    #print("yasssssh")
    #print(tempsoup)
    ques=None
    ans=None
    whetherTable=None
    if tempsoup:
        #print("yash")
        #article_mcq=tempsoup.find('div',class='box-section online-quiz clip-last-child thm-bg')
        div_element = tempsoup.find('div', class_='box-section online-quiz clip-last-child thm-bg')
        if div_element==None:
            return None,ques,ans
        #print(div_element)
        # Find the question (h4) element
        question_element = div_element.find('h4', class_='box-section--title')

        # Find all the p elements within the div (answers)
        p_elements = div_element.find_all('p', class_='para')

        # Extract and print the question and answers
        question = question_element.text
        answers = [p.text for p in p_elements]
        
        # print("Question:", question)
        # print("Answers:")
        # for answer in answers:
        #     print(answer)
        whetherTable=1
        ques=question
        ans=answers
        return whetherTable,ques,answers



baba=0
for ind in dataframe1.index:
    if ind==1865:
        break
    print("index: ",ind)
    url1=dataframe1['Key'][ind]
    time.sleep(random.uniform(5, 10))
    scraper = cloudscraper.create_scraper(delay=20, browser="chrome") 
    content = scraper.get(url1).text
    soup = BeautifulSoup(content, 'html.parser')
    #print(soup)
    results=soup.findAll("div",{"class":"article-content"})
    #print(results[0])
    # #print(results)

    #successfully extracted title
    #title_element = soup.find('h1', class_='meta-article-title')
    titlevalue=dataframe1['Value'][ind]
    print("Title: ",titlevalue)
    checkimage=False


    whethermcq,question,answers=extractMCQ(soup)
    # print("baabbaa")
    if whethermcq==None:
        print("No MCQ found or some other issue....trying again ")
        #again try to fetch soup 
        time.sleep(random.uniform(15, 30))
        scraper = cloudscraper.create_scraper(delay=20, browser="chrome")
        content = scraper.get(url1).text
        soup = BeautifulSoup(content, 'html.parser')
        whethermcq,question,answers=extractMCQ(soup)

        if whethermcq==None:
            print("Again No MCQ found...Move to next")
            continue

    paragraphs,diagnosis,chooseoption,casepara,discussionpara = extract_paragraphs(soup)

    checkImage=hasImage(soup)
    HasImage="No"

    if checkImage==True:
        HasImage="Yes"
    
    articleType,superclass=tellfield(soup)



    # print("Question:", question)
    # print("Answers:")
    for answer in answers:
        print(answer)
    # print("Diagnosis: ",diagnosis)
    # print("chooseoption: ",chooseoption)

    combineCasepara=""
    combinediscussionpara=""

    for para in casepara:
        combineCasepara+=para
    
    for para in discussionpara:
        combinediscussionpara+=para

    dff={'URL' : url1, 'Title' : titlevalue, 'Case' : combineCasepara,'Discussion' : combinediscussionpara,'MCQ_question': question,'Option1': answers[0],'Option2': answers[1],'Option3': answers[2],'Option4': answers[3],'Diagnosis': diagnosis,'Correct_option': chooseoption,'HasImage': HasImage,'MedicalField': articleType,'Superclass': superclass}
    baba=baba+1
    print("Successfully_fetched",baba)
    df = df.append(dff, ignore_index = True)


df.to_excel("output_fin_jama_big2.xlsx")
sys.stdout = sys._stdout_
file.close()