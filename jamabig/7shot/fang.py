import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as palm
import torch
import json
import os
import pandas as pd
import csv
import argparse
from tqdm import tqdm
import together
import random

openai.api_key = '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'
token = "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"
palm.configure(api_key='AIzaSyAILWnvzeDIDPrOuzMahvRPUG7RTniujv8')

together.api_key = '74c6f051adb52d62603950542c58dc556c8a38abb68a96336e9cdd0210d46e01'

# construct inputs
def input_format(args, examples, prompt_type=None):
    inputs = []
    fewshot_examples = []
    if prompt_type == 'YR':
        for example in examples:
            question = example['question']
            opa = example['opa']
            opb = example['opb']
            opc = example['opc']
            opd = example['opd']
            ope = example['ope']
            if 'palm' in args.model_name:
                inputs.append(
                    f"{question} A: {opa}, B: {opb}, C: {opc}, D: {opd}, E: {ope}. Please choose an answer and explain why that might be correct while the rest are incorrect. The output format is: Answer: (fill in the letter of the answer) Explanation: "
                )
            else:
                inputs.append(
                    f"Given the following clinical case, please choose an answer and explain why that might be correct while the rest are incorrect. The output format is: \nAnswer: (fill in the letter of the answer) \nExplanation: \n{question} A: {opa}, B: {opb}, C: {opc}, D: {opd}, E: {ope}."
                    # f"{question} A: {opa}, B: {opb}, C: {opc}, D: {opd}, E: {ope}. Please choose an answer and explain why that might be correct while the rest are incorrect. The output format is: Answer: (fill in the letter of the answer) Explanation: "
                )
    elif prompt_type == 'R':
        for example in examples:
            question = example['question']
            opa = example['opa']
            opb = example['opb']
            opc = example['opc']
            opd = example['opd']
            answer_idx = example['answer_idx']
            # answer = example['answer']
            
            if args.option_num==5:
                ope = example['ope']
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: \"A\": {opa}, \"B\": {opb}, \"C\": {opc}, \"D\": {opd}, \"E\": {ope}\n"\
                    f"ANSWER:{answer_idx}\n"\
                    f"Q:\"You are a large language model that just answered the above question. Please explain why {answer_idx} is correct answer while the rest choices are incorrect. You should explain each choice in detail.\"\n"\
                    "A:"
                )
            elif args.option_num==4:
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: \"A\": {opa}, \"B\": {opb}, \"C\": {opc}, \"D\": {opd}\n"\
                    f"ANSWER:{answer_idx}\n"\
                    f"Q:\"You are a large language model that just answered the above question. Please explain why {answer_idx} is correct answer while the rest choices are incorrect. You should explain each choice in detail.\"\n"\
                    "A:"
                )
            if args.few_shot != 0:
                fewshot_exps = []
                current_path = os.path.dirname(os.path.abspath(__file__))
                holdout_file = os.path.join(current_path, 'data', args.data_name, "holdout_medbullets_op"+str(args.option_num)+".csv")
                holdout_df = pd.read_csv(holdout_file)
                cnt = 0
                for index,row in holdout_df.iterrows():
                    cnt += 1
                    if args.option_num == 5:
                        case = f"QUESTION: {row['question']}\n"\
                        f"ANSWER CHOICES: \"A\": {row['opa']}, \"B\": {row['opb']}, \"C\": {row['opc']}, \"D\": {row['opd']}, \"E\": {row['ope']}\n"\
                        f"ANSWER:{row['answer_idx']}\n" \
                        f"Q:\"You are a large language model that just answered the above question. Please explain why {row['answer_idx']} is correct answer while the rest choices are incorrect. You should explain each choice in detail.\"\n"\
                        "A:"
                    elif args.option_num == 4:
                        case = f"QUESTION: {row['question']}\n"\
                        f"ANSWER CHOICES: \"A\": {row['opa']}, \"B\": {row['opb']}, \"C\": {row['opc']}, \"D\": {row['opd']}\n"\
                        f"ANSWER:{row['answer_idx']}\n" \
                        f"Q:\"You are a large language model that just answered the above question. Please explain why {row['answer_idx']} is correct answer while the rest choices are incorrect. You should explain each choice in detail.\"\n"\
                        "A:"
                    explanation = row['explanation']
                    fewshot_exps.append((case, explanation))
                    if cnt >= args.few_shot:
                        break
                fewshot_examples.append(fewshot_exps)
            
    elif prompt_type == 'RY': # CoT
        for example in examples:
            question = example['question']
            opa = example['opa']
            opb = example['opb']
            opc = example['opc']
            opd = example['opd']
            # answer_idx = example['answer_idx']
            if args.option_num == 5:
                ope = example['ope']
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: \"A\": {opa}, \"B\": {opb}, \"C\": {opc}, \"D\": {opd}, \"E\": {ope}\n"\
                    f"Let's think step by step. You should analyze each choice in detail.\n"\
                )
            elif args.option_num == 4:
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: \"A\": {opa}, \"B\": {opb}, \"C\": {opc}, \"D\": {opd}\n"\
                    f"Let's think step by step. You should analyze each choice in detail.\n"\
                )
    elif prompt_type == 'Y':
        if args.few_shot != 0:
            for idx, example in enumerate(examples):
                idxes = [i for i in range(len(examples))]
                idxes.remove(idx)
                select_idx = random.sample(idxes, k=args.few_shot)
                fewshot_exps = []
                for id in select_idx:
                    if args.option_num == 5:
                        if 'palm' in args.model_name:
                            case = f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {examples[id]['question']} (A) {examples[id]['opa']} (B) {examples[id]['opb']} (C) {examples[id]['opc']} (D) {examples[id]['opd']} (E) {examples[id]['ope']} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        else:
                            case = f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {examples[id]['question']} (A) {examples[id]['opa']} (B) {examples[id]['opb']} (C) {examples[id]['opc']} (D) {examples[id]['opd']} (E) {examples[id]['ope']}"
                        answer = examples[id]['answer_idx']
                    else:
                        if 'palm' in args.model_name:
                            case = f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {examples[id]['question']} (A) {examples[id]['opa']} (B) {examples[id]['opb']} (C) {examples[id]['opc']} (D) {examples[id]['opd']} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        else:
                            case = f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {examples[id]['question']} (A) {examples[id]['opa']} (B) {examples[id]['opb']} (C) {examples[id]['opc']} (D) {examples[id]['opd']}"
                        answer = examples[id]['answer_idx']
                    fewshot_exps.append((case, answer))
                fewshot_examples.append(fewshot_exps)
                question = example['question']
                opa = example['opa']
                opb = example['opb']
                opc = example['opc']
                opd = example['opd']
                if args.option_num == 5:
                    ope = example['ope']
                    if 'palm' in args.model_name:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        )
                    else:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}"
                        )
                else:
                    if 'palm' in args.model_name:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        )
                    else:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd}"
                        )
        else:
            for example in examples:
                question = example['question']
                opa = example['opa']
                opb = example['opb']
                opc = example['opc']
                opd = example['opd']
                if args.option_num == 5:
                    ope = example['ope']
                    if 'palm' in args.model_name:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        )
                    else:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}"
                        )
                else:
                    if 'palm' in args.model_name:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} \nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        )
                    else:
                        inputs.append(
                            f"The following are multiple choice questions (with answers) about medical knowledge. **Question:** {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd}"
                        )
    
    return inputs, fewshot_examples

# prompt LLMs
def call_gpt(message, args, fewshot_exps=None):
    if args.prompt_type == 'RY':
        messages = [ {"role": "system", "content": 
                    "You are a helpful assistant that good at dealing with multiple-choice medical questions."} ]
        messages.append(
                {"role": "user", "content": message},
            )
        if args.use_chat:
            response = openai.ChatCompletion.create(
                model=args.model_name, messages=messages
            )
            output = response.choices[0].message.content
        else:
            response = openai.Completion.create(
                model=args.model_name,
                prompt=message
            )
            output = response.choices[0]["text"]
        messages.append(
            {"role": "assistant", "content": output},
        )
        messages.append(
            {"role": "user", "content": f'Therefore, accoding to your analysis, the only correct answer is ('},
        )
        input_cot = message + '\n' + output + '\nTherefore, accoding to your analysis, the only correct answer is ('
        if args.use_chat:
            response = openai.ChatCompletion.create(
                model=args.model_name, messages=messages
            )
            output_cot = response.choices[0].message.content
        else:
            response = openai.Completion.create(
                model=args.model_name,
                prompt=message
            )
            output_cot = response.choices[0]["text"]
    else:
        if args.prompt_type == 'Y':
            messages = [ {"role": "system", "content": 
                    "You are a helpful assistant that answers multiple choice questions about medical knowledge."} ]
            if fewshot_exps:
                for exp in fewshot_exps:
                    messages.append(
                    {"role": "user", "content": exp[0]},
                    )
                    messages.append(
                        {"role": "assistant", "content": "**Answer**:("+exp[1]+")"},
                    )
            messages.append(
                {"role": "user", "content": message},
            )
            messages.append(
                {"role": "assistant", "content": "**Answer**:("},
            )
        elif args.prompt_type == 'R':
            messages = [ {"role": "system", "content": 
                    "You are a helpful assistant that explains the answer of multiple-choice medical questions."} ]
            if fewshot_exps:
                for exp in fewshot_exps:
                    messages.append(
                    {"role": "user", "content": exp[0]},
                    )
                    messages.append(
                        {"role": "assistant", "content": exp[1]},
                    )
            messages.append(
                {"role": "user", "content": message},
            )
        if args.use_chat:
            response = openai.ChatCompletion.create(
                model=args.model_name, messages=messages
            )
            output = response.choices[0].message.content
        else:
            response = openai.Completion.create(
                model=args.model_name,
                prompt=message
            )
            output = response.choices[0]["text"]
        input_cot = ''
        output_cot = ''
    return output, input_cot, output_cot

# def call_llama2(message, tokenizer, model):
#     inputs = tokenizer.encode(
#         message,
#         return_tensors="pt"
#     )

#     outputs = model.generate(
#         inputs,
#     )
#     output = tokenizer.decode(outputs[0])
        
#     return output

def call_llama2_togetherai(message, args, fewshot_exps=None):
    messages = ""
    model_name = "togethercomputer/" + args.model_name
    if args.prompt_type == 'RY':
        messages += f"<HUMAN>: {message}\n<ROBOT>: "
        response = together.Complete.create(
                    prompt = messages,
                    model = model_name,
                    max_tokens = 1024,
                    temperature = 0.8,
                    top_k = 60,
                    top_p = 0.95,
                    repetition_penalty = 1.1,
                    # stop = ['<human>', '\n\n']
                )
        output = response['output']['choices'][0]['text'].strip()
        messages += output + '\nTherefore, accoding to my analysis, the only correct answer is ('
        input_cot = messages

        response = together.Complete.create(
                    prompt = messages,
                    model = model_name,
                    max_tokens = 1024,
                    temperature = 0.8,
                    top_k = 60,
                    top_p = 0.95,
                    repetition_penalty = 1.1,
                    # stop = ['<human>', '\n\n']
                )
        output_cot = response['output']['choices'][0]['text'].strip()
    else:
        if args.prompt_type=='Y':
            if fewshot_exps:
                for exp in fewshot_exps:
                    messages += f"<human>: {exp[0]}\n<bot>:"
                    messages += "**Answer**:("+exp[1]+")\n"
                messages += f"<human>: {message}\n<bot>: "
            else:
                messages += f"<human>: {message}. Please choose an answer, strictly following the output format '**Answer**:(fill in the letter of the answer)'\n<bot>:"
        elif args.prompt_type=='R':
            if fewshot_exps:
                for exp in fewshot_exps:
                    messages += f"<HUMAN>: {exp[0]}\n"
                    messages += f"<ROBOT>: {exp[1]}\n"
            messages += f"<HUMAN>: {message}\n<ROBOT>: "
        response = together.Complete.create(
                    prompt = messages,
                    model = model_name,
                    max_tokens = 1024,
                    temperature = 0.8,
                    top_k = 60,
                    top_p = 0.95,
                    repetition_penalty = 1.1,
                    # stop = ['<human>', '\n\n']
                )
        output = response['output']['choices'][0]['text'].strip()
        input_cot = ''
        output_cot = ''
    return output, input_cot, output_cot

# def call_meditron(message, tokenizer, model, args):
#     inputs = tokenizer.encode(
#         message,
#         return_tensors="pt"
#     )
#     inputs = inputs.to(args.device)

#     outputs = model.generate(
#         inputs,
#     )
#     output = tokenizer.decode(outputs[0])
        
#     return output

def call_palm(message, args, fewshot_exps=None):
    examples = None
    messages = []
    input_cot = ''
    output_cot = ''
    if args.prompt_type == 'RY':
        messages.append(
            {"author": "user", "content": message},
        )
        # defalut chat mode
        response=palm.chat(
                        model='models/chat-bison-001',
                        examples=examples,
                        messages=messages, 
                        temperature=0.8, 
                        context="You are a helpful assistant that good at dealing with multiple-choice medical questions."
                    )
        try:
            output = response.candidates[0]['content']
        except:
            output = ' '
        messages.append(
            {"author": "assitant", "content": output},
        )
        messages.append(
            {"author": "user", "content": 'Therefore, please choose an answer accoding to your analysis, strictly following the output format \'Answer:(fill in the letter of the answer without eliminating the brackets)\' '},
        )
        input_cot = message + '\n' + output + '\nTherefore, please choose an answer accoding to your analysis, strictly following the output format \'Answer:(fill in the letter of the answer)\' '

        response=palm.chat(
                        model='models/chat-bison-001',
                        examples=examples,
                        messages=messages, 
                        temperature=0.8, 
                        context="You are a helpful assistant that good at dealing with multiple-choice medical questions."
                    )
        try:
            output_cot = response.candidates[0]['content']
        except:
            output_cot = ''
    else:
        if args.prompt_type == 'Y':
            if fewshot_exps:
                examples = []
                for exp in fewshot_exps:
                    examples.append(
                            { 
                            "input": {"content": exp[0]},
                            "output": {"content": "Answer:("+exp[1]+")"}
                        }
                    )
        elif args.prompt_type == 'R':
            if fewshot_exps:
                examples = []
                for exp in fewshot_exps:
                    examples.append(
                            { 
                            "input": {"content": exp[0]},
                            "output": {"content": exp[1] }
                        }
                    )
        messages.append(
            {"author": "user", "content": message},
        )
        if args.use_chat:
            #--------------------Problem: inconsistent output formats---------------------
            response=palm.chat(
                            model='models/chat-bison-001',
                            examples=examples,
                            messages=messages, 
                            temperature=0.8, 
                            context="You are a helpful assistant that answers multiple choice questions about medical knowledge."
                        )
            try:
                output = response.candidates[0]['content']
            except:
                output = ''
        else:
            #--------------------Problem: most outputs are none---------------------
            prompt = ""
            if fewshot_exps:
                for exp in fewshot_exps:
                    prompt += exp[0]
                    prompt += "("+exp[1]+")\n"
            prompt += message
            # prompt = "The following are multiple choice questions (with answers) about medical knowledge. Question: A 27-year-old man presents for an appointment to establish care. He recently was released from prison. He has felt very fatigued and has had a cough. He has lost roughly 15 pounds over the past 3 weeks. He attributes this to intravenous drug use in prison. His temperature is 99.5¬∞F (37.5¬∞C), blood pressure is 127/68 mmHg, pulse is 100/min, respirations are 18/min, and oxygen saturation is 98% on room air. QuantiFERON gold testing is positive. The patient is started on appropriate treatment. Which of the following is the most likely indication to discontinue this patient's treatment? A: Elevated liver enzymes, B: Hyperuricemia, C: Optic neuritis, D: Peripheral neuropathy, E: Red body excretions. Answer:"
            response = palm.generate_text(
                            model='models/text-bison-001',
                            prompt=prompt,
                            temperature=0.8,
                            max_output_tokens=512,
                        )
            output = response.result
    return output, input_cot, output_cot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_type",
        default='Y',
        type=str,
        help="YR: output answer and explanation; R: output explanation given correct answer; RY: chain-of-though; Y: answer",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="model name: gpt-3.5-turbo, gpt-4, llama-2-70b-chat, palm2, gpt-3.5-turbo-instruct, meditron",
    )
    parser.add_argument(
        "--device", default="gpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--data_name",
        default="medbullets",
        type=str,
        help="data name: medqa, medbullets",
    )
    parser.add_argument(
        "--input_file",
        default="medbullets_test.csv",
        type=str,
        help="input file: medbullets_test.csv, medbullets_test4.csv, test.csv, test5.csv",
    )
    parser.add_argument(
        "--prompt_file",
        default="medbullets_prompt.csv",
        type=str,
        help="prompt file",
    )
    parser.add_argument(
        "--output_file",
        default="results",
        type=str,
        help="output file",
    )
    parser.add_argument(
        "--use_chat",
        default=False,
        action="store_true",
        help="Use the chat version.",
    )
    parser.add_argument(
        "--few_shot",
        default=0,
        type=int,
        help="Number of few-shot examples.",
    )
    parser.add_argument(
        "--option_num",
        default=5,
        type=int,
        help="Number of options.",
    )
    args = parser.parse_args()

    print('model_name: {}'.format(args.model_name))
    print('prompt_type: {}'.format(args.prompt_type))

    # Setup device
    args.device = torch.device(
        f"cuda"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    
    # read data
    current_path = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(current_path, 'data', args.data_name, args.input_file)
    header = pd.read_csv(in_file, nrows = 0).columns.to_list()
    df = pd.read_csv(in_file)
    examples = []
    # ii = 0
    for _, row in df.iterrows():
        # ii += 1
        # if ii == 10:
        #     break
        examples.append(row)

    inputs, fewshot_examples = input_format(args, examples, args.prompt_type)
    outputs = []
    inputs_cot = []
    outputs_cot = []
    if 'llama-2' in args.model_name:
        # load model from huggingface
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "meta-llama/"+args.model_name
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     "meta-llama/"+args.model_name,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        # )

        # outputs = []
        # for input in tqdm(inputs):
        #     output = call_llama2(input, tokenizer, model)
        #     outputs.append(output)

        # together ai
        if len(fewshot_examples) != 0:
            for input, fewshot_exps in tqdm(zip(inputs, fewshot_examples)):
                output,input_cot,output_cot = call_llama2_togetherai(input, args, fewshot_exps)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)
        else:
            for input in tqdm(inputs):
                output,input_cot,output_cot = call_llama2_togetherai(input, args)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)

    elif 'gpt' in args.model_name:  
        # call model
        if len(fewshot_examples) != 0:
            for input, fewshot_exps in tqdm(zip(inputs, fewshot_examples)):
                output,input_cot,output_cot = call_gpt(input, args, fewshot_exps)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)
        else:
            for input in tqdm(inputs):
                output,input_cot,output_cot = call_gpt(input, args)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)
    
    elif 'palm' in args.model_name:
        if len(fewshot_examples) != 0:
            for input, fewshot_exps in tqdm(zip(inputs, fewshot_examples)):
                output,input_cot,output_cot = call_palm(input, args, fewshot_exps)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)
        else:
            for input in tqdm(inputs):
                output,input_cot,output_cot = call_palm(input, args)
                outputs.append(output)
                inputs_cot.append(input_cot)
                outputs_cot.append(output_cot)

    # elif 'meditron' in args.model_name:
    #     # load model from huggingface
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "epfl-llm/meditron-70b"
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(
    #         "epfl-llm/meditron-70b",
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #     )
    #     model.to(args.device)

    #     for input in tqdm(inputs):
    #         output = call_meditron(input, tokenizer, model, args)
    #         outputs.append(output)

    # write into file
    filename = args.input_file.split('.')[0]
    output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'.csv')
    if args.few_shot != 0:
        output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'_fs_'+\
                               str(args.few_shot)+'.csv')
    combs = []
    for e, i, o, i_cot, o_cot in zip(examples[:len(inputs)], inputs, outputs, inputs_cot, outputs_cot):
        comb = [e[col] for col in header]
        comb.append(i)
        comb.append(o)
        comb.append(i_cot)
        comb.append(o_cot)
        combs.append(comb)
    header.append('input')
    header.append('output')
    header.append('input_cot')
    header.append('output_cot')
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combs)


if __name__ == "__main__":
    main()