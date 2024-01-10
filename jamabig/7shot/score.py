import os
import csv
import pandas as pd
import re
import argparse
from rouge import Rouge
from bert_score import BERTScorer
from bart_score import BARTScorer
from tqdm import tqdm

def replace_numbers_with_letters(match):
    # Extract the numbers from the match
    numbers = match.group(1)

    # Convert each digit to a corresponding letter
    letters = ''.join(chr(int(digit) + ord('A') - 1) for digit in numbers)

    # Reconstruct the modified substring
    return f"Answer {letters}:"

def replace_substring(input_string):
    # Define the regular expression pattern
    pattern = r'Answer (\d+|\d+-\d+):'

    # Use re.sub() to replace matched substrings
    result_string = re.sub(pattern, replace_numbers_with_letters, input_string)

    return result_string

def compute_score(gold_exp, pred_exp, metric):
    if metric == 'rouge':
        try:
            rouge_scorer = Rouge()
            score = rouge_scorer.get_scores(pred_exp, gold_exp)[0]["rouge-l"]["f"]
        except:
            score = 0
    elif metric == 'bertscore':
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([pred_exp], [gold_exp])
        score = float(F1.mean())
    elif metric == 'bartscore_cnn':
        bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
        score = bart_scorer.score([gold_exp], [pred_exp], batch_size=4)
        score = float(score[0])
    elif metric == 'bartscore_cnn_para':
        bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
        bart_scorer.load(path='./scoreckpt/bart_score.pth')
        score = bart_scorer.score([gold_exp], [pred_exp], batch_size=4)
        score = float(score[0])

    return score

def score_func(gold_exp, output, metric):
    pred_exp = ""
    # replace numbers with letters
    gold_exp = replace_substring(gold_exp)
    try:
        output = re.sub("Answer:\n", "Answer: ", output)
        pred_exp = re.sub(r'Answer:[^\n]*\n', '', output)
        pred_exp = re.sub("Explanation: ", "Explanation:", pred_exp)
        pred_exp = re.sub('Explanation:', '', pred_exp)
        score = compute_score(gold_exp, pred_exp, metric)
    except:
        score = compute_score(gold_exp, pred_exp, metric)
    return gold_exp, pred_exp, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_type",
        default='YR',
        type=str,
        help="YR: output answer and explanation; R: output explanation given correct answer; RY: chain-of-though; Y: answer",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="model name: gpt-3.5-turbo, llama-2-70b-chat, gpt-3.5-turbo-instruct, palm2, gpt-4",
    )
    parser.add_argument(
        "--data_name",
        default="medbullets",
        type=str,
        help="data name",
    )
    parser.add_argument(
        "--input_file",
        default="total_medbullets_op5.csv",
        type=str,
        help="input file: total_medbullets_op4.csv, total_medbullets_op5.csv",
    )
    parser.add_argument(
        "--output_file",
        default="explanations",
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
        "--metric",
        default="rouge",
        type=str,
        help="rouge, bertscore, bartscore_cnn, bartscore_cnn_para",
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

    print('prompt_type: {}'.format(args.prompt_type))
    print('model_name: {}'.format(args.model_name))
    print('metric: {}'.format(args.metric))

    # read data
    current_path = os.path.dirname(os.path.abspath(__file__))
    filename = args.input_file.split('.')[0]
    in_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'.csv')
    header = pd.read_csv(in_file, nrows = 0).columns.to_list()
    df = pd.read_csv(in_file)
    examples = []
    for _, row in df.iterrows():
        examples.append(row)

    scores = []
    combs = []
    for example in tqdm(examples):
        gold_exp = example['explanation']
        output = example['output']
        gold_exp, pred_exp, score =  score_func(gold_exp, output, args.metric)
        scores.append(score)
        comb = [example[col] for col in header]
        comb.append(gold_exp)
        comb.append(pred_exp)
        comb.append(score)
        combs.append(comb)
    header.append('gold_explanation')
    header.append('pred_explanation')
    header.append('score')

    print('{}: {}'.format(args.metric, sum(scores) / len(scores)))
    output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type \
                               +'_exp'+'_'+args.metric+'.csv')
    if args.few_shot != 0:
        output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'_fs_'+\
                               str(args.few_shot)+'_exp'+'_'+args.metric+'.csv')
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combs)


if __name__ == "__main__":
    main()