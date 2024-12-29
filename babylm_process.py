from tokenizers import ByteLevelBPETokenizer
import re
import glob
import chardet
import nltk
from nltk import Tree
nltk.download('punkt')
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import spacy
import benepar
benepar.download('benepar_en3')
from benepar.spacy_plugin import BeneparComponent
import argparse
import torch

if torch.cuda.is_available():
    spacy.prefer_gpu()

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", required=True, type=str, help="File that leads to a corpus to parse.")

def break_paragraph(sent):
    new_sents = []
    for s in sent.split(". "):
        new_sents.append(s + ".")
    return new_sents

def parse_corpus(nlp, corpus):
    file_name = corpus.split('/')
    corpus_name = file_name[-1]

    with open(corpus, 'r') as f:
        content = f.readlines()
    
    outfile = f'babylm_data/dev_parses/parsed_{corpus_name}'
    skip_file = f'babylm_data/skipped/{corpus_name}'
    with open(outfile, 'w') as output, open(skip_file, 'w') as skip:
        for i, line in enumerate(content):
            print(i+1)
            line = line.replace('\n', '').replace('\t', '').replace('.  ', '. ').replace('!  ', '! ').replace('?  ', '? ')
            sep_sents = break_paragraph(line)
            for s in sep_sents:
                try:
                    parsed = nlp(s)
                    for sent in parsed.sents:
                        parse = sent._.parse_string
                        output.write(parse + "\n")
                except Exception as e:
                    skip.write(s + '\n')
                    continue
                    
def main(args):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    
    file = args.corpus_file
    parse_corpus(nlp, file)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)