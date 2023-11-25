import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as palm
import torch
import os
import pandas as pd
import csv
import argparse
from tqdm import tqdm

token = "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"

dataframe1 = pd.read_excel('output_fin_jama_big2.xlsx')
dataframe2=dataframe1

dataframe2['message_prompt1']=""
dataframe2['message_prompt2']=""
dataframe2['Answer_llama']=""
dataframe2['Explanation']=""
dataframe2['Actual_Correct_option']=""
dataframe2['IsLlamaCorrect']=""
dataframe2['ModifiedMedicalField']=""