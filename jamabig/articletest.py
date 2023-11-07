import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random
import subprocess

#dataframe1 = pd.read_excel('outputjama_big.xlsx')

dic={}


df = pd.DataFrame(columns = ['URL', 'Title','Case','Discussion','MCQ_question','Option1','Option2','Option3','Option4','Diagnosis','Correct_option','HasImage','MedicalField'])

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
    print("in extract para")
    if tempsoup:
        print("in temp soup")
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


def tellfield(tempsoup):
    article_para = tempsoup.find('div', class_='meta-article-type thm-col')
    super_class = tempsoup.find('div', class_='meta-super-class')
    #print("Journal: ",article_para.get_text())
    print(super_class.get_text())
    return article_para.get_text()
    

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
# for ind in dataframe1.index:
#     if ind==1764:
#         break
#     print("index: ",ind)
url1="https://jamanetwork.com/journals/jamaotolaryngology/fullarticle/2810850"
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
titlevalue="Generalized, Linear Hyperkeratotic Papules With Numerous White Spicules and Comedo-like Plugs"
print("Title: ",titlevalue)
checkimage=False


whethermcq,question,answers=extractMCQ(soup)
# print("baabbaa")
if whethermcq==None:
    sys.exit(0)
paragraphs,diagnosis,chooseoption,casepara,discussionpara = extract_paragraphs(soup)

checkImage=hasImage(soup)
HasImage="No"

if checkImage==True:
    HasImage="Yes"

articleType=tellfield(soup)



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

dff={'URL' : url1, 'Title' : titlevalue, 'Case' : combineCasepara,'Discussion' : combinediscussionpara,'MCQ_question': question,'Option1': answers[0],'Option2': answers[1],'Option3': answers[2],'Option4': answers[3],'Diagnosis': diagnosis,'Correct_option': chooseoption,'HasImage': HasImage,'MedicalField': articleType}
baba=baba+1
print("Successfully_fetched",baba)
df = df.append(dff, ignore_index = True)

df.to_excel("output_check.xlsx")
#print(df)