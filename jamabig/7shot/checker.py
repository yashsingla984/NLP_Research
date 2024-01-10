import argparse
import csv
import cloudscraper 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests import get
import numpy as np
import time
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as palm
import torch
from tqdm import tqdm
import together

import openai 
import openpyxl

in_file='input_jama2.xlsx'
df = pd.read_excel(in_file)

examples = []
ii = 0
count=0
for _, row in df.iterrows():
    ii += 1
    actual_pattern = re.compile(r'\b([A-D]).', re.DOTALL)
    actual=actual_pattern.search(row['Correct_option'])
    if actual is None:
        print(ii)
        count=count+1
        print(actual)
    else:
        examples.append(row)

print(ii)
print(count)