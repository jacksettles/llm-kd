import json
import jsonlines
import os
import glob
from tokenizers import ByteLevelBPETokenizer
import torch
import torch.nn.functional as F
from gpt2_model import GPT, generate_square_subsequent_mask
import pandas as pd
import argparse

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available. Using GPU.")
else:
    device = "cpu"
    print("Cuda is not available. Using CPU.")
    
parser = argparse.ArgumentParser()
    
parser.add_argument("--tokenizer_vocab", default="tokenizers/rnng/vocab.json", type=str,
                    help="File location for the vocab.json to a tokenizer")
parser.add_argument("--tokenizer_merges", default="tokenizers/rnng/merges.txt", type=str,
                    help="File location for the merges.txt to a tokenizer")
parser.add_argument("--control_model", default="saved_models/bllip/solo/bllip_ltg_gpt2.pt", type=str, help="Base model for comparison.")
parser.add_argument("--distilled_model", default="saved_models/bllip/distilled/distilled_bllip_ltg_gpt2.pt", type=str,
                    help="Distilled model to test against control model.")
parser.add_argument("--savepath", default="eval_results/blimp/", type=str, help="Location to save scores")

def get_probs(sentence, model, tokenizer):
    with torch.no_grad():
        tokens = tokenizer.encode(sentence).ids

        # Add BOS and EOS tokens to match training regimen
        tokens.insert(0, 0)
        tokens.append(2)

        # Add batch dimension and move to device
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        model.to(device)
        inputs = tokens[:, :-1]
        labels = tokens[:, 1:]

        mask = generate_square_subsequent_mask(size=inputs.size(1), device=device)
        logits = model(input_ids=inputs, attention_mask=mask)
        log_probs_word = F.log_softmax(logits, dim=-1)

        # Add dimension to labels
        # Then gather lob probs in log_probs_word based on values in labels
        # Then ditch the last dimension and sum total log_probs
        gathered_log_probs = torch.gather(log_probs_word, 2, labels.unsqueeze(2)).squeeze(2).sum(1)
        return gathered_log_probs
    
def run_test_suite(model, files, tokenizer, seed=42):
    torch.manual_seed(seed)
    score_dict = {}
    i = 0
    for file in files:
        with jsonlines.open(file) as reader:
            data = [obj for obj in reader]
        total_sents = len(data)
        total_correct = 0
        for test in data:
            good_sentence = " " + test['sentence_good']
            bad_sentence = " " + test['sentence_bad']

            good_probs = get_probs(good_sentence, model, tokenizer)
            bad_probs = get_probs(bad_sentence, model, tokenizer)

            if good_probs > bad_probs:
                total_correct += 1
        score = (total_correct / total_sents)*100
        score_dict[file] = score
        i += 1
        print(f'{i}:\t{file}\t- Score: {score}')
    return score_dict

def save_args(args, output_dir, model_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the path for the JSON file
    json_file_path = os.path.join(output_dir, f'{model_name}_args.json')
    
    # Save the argument dictionary as a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(args, json_file, indent=4)

def main(args):
    tokenizer = ByteLevelBPETokenizer(args.tokenizer_vocab, args.tokenizer_merges)
    
    # Gather all the BLiMP test files
    pattern = "blimp_data/*.jsonl"
    blimp_files = glob.glob(pattern, recursive=True)
    
    # Set up models and their args
    c_load = torch.load(args.control_model)
    d_load = torch.load(args.distilled_model)
    
    control_model = c_load['model']
    distilled_model = d_load['model']
    control_args = c_load['args']
    distilled_args = d_load['args']
    
    # Evaluate models
    control_scores = run_test_suite(control_model, blimp_files, tokenizer)
    distilled_scores = run_test_suite(distilled_model, blimp_files, tokenizer)
    
    # Log results
    control_df = pd.DataFrame([control_scores], index=[0]).T
    dist_df = pd.DataFrame([distilled_scores], index=[0]).T
    control_df.columns = ["Control"]
    dist_df.columns = ["Distilled"]
    
    # Save model args so we know exactly which models were tested
    save_args(control_args, args.savepath, "sequential_ltg_bllip")
    save_args(distilled_args, args.savepath, "distilled_ltg_bllip")
    
    # Save scores as .csv file
    save_path = args.savepath + "bllip_ltg_test_scores.csv"
    master_df = pd.concat([control_df, dist_df], axis=1)
    master_df.to_csv(save_path, index=True)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)