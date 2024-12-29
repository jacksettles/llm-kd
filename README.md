# Knowledge Distillation for LLMs

This is an implementation of my master's thesis. It borrows and modifies
code from the github repository associated with this paper:  
[Unsupervised Recurrent Neural Network Grammars](https://arxiv.org/abs/1904.03746)  
Yoon Kim, Alexander Rush, Lei Yu, Adhiguna Kuncoro, Chris Dyer, Gabor Melis  
NAACL 2019  
Github repo: https://github.com/harvardnlp/urnng

## Dependencies
The code was tested in `python 3.10.4` and `pytorch 2.1.2`.

## Data  
Multiple datasets were used for this project, namely:
 - The Penn Treebank (PTB): Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a Large Annotated Corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330.
 - BLLIP Corpus
 - BabyLM Corpus: Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell. 2023. Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 1–34, Singapore. Association for Computational Linguistics.
 
 
PTB and BLLIP are already parsed, with PTB being gold standard parses (done manually by a human), but only about 1 million words. BLLIP contains silver grade parses (done by a machine parser), but contains about 30 million words. The BabyLM corpus had to be parsed using the SpaCy implementation of the Berkely neural parser. This can be seen in the `babylm_process.py` file.

Once all data was parsed, the words within the leaves of each tree had to be tokenized. An example of how this was done can be seen in the `tokenize_data.py` file within the babylm_data subdirectory.

Once you have parsed and tokenized sentences, they need to be split into train.txt, val.txt (or dev.txt, your preference), and test.txt files.
Then you can preprocess the data using the `preprocess_tokenized_data.py` file. An example script with the full command that runs this file can be found in `scripts/process_tokenized_data.sh`. This is a modified version of the urnng `preprocess.py` script, which had to be changed to handle tokenized data. The main change here was that an additional parsing SHIFT action was added for every token, rather than every word. It was also modified to take in a predefined vocabulary. This is crucial because the RNNG you use as the teacher in your KD setup has to have the same vocabulary, and same output layer index to word mapping, as the student transformer LLM. The student vocabulary is defined by the tokenizer you train on your data. This tokenizer is trained on the raw text, not the parsed-tokenized text. For this project, we used a vocab of 12000 tokens.

Once you run this preprocessing script, you should have some .pkl files for these train, val, and test splits, plus a .dict. This is for the RNNG. These files get passed into RNNGDataset objects (just renamed from the urnng implementation to avoid confusion with the PyTorch Dataset class). When we trained our transformer LLM, we needed to use multiple GPUs to split up the data. To do this seemlessly and take advantage of PyTorch's Dataset, DataLoader and DistributedSampler modules, we made a custom GPT2Dataset class. This can be passed to a DataLoader, which can take a DistributedSampler, as seen in the `multi_gpu_gpt2_train.py` file.

## Training

To train the RNNG on 1 GPU:
```
python train_tokenized_rnng.py --train_file babylm_data/tokenized/babylm_final_dataset-train.pkl --val_file babylm_data/tokenized/babylm_final_dataset-val.pkl --save_path saved_models/rnng/babylm_tokenized_rnng.pt --mode supervised --train_q_epochs 18 --decay 0.5 --lr 1 --dropout 0.5 --gpu 0
```
This can take a few weeks if using 1 GPU and the full BabyLM Dataset. BLLIP will also take a few weeks, as these are cumbersome to train.

To train the transformer LLM with KD:
```
torchrun --standalone --nproc_per_node=gpu multi_gpu_gpt2_train.py --ff_dim 768 --num_heads 4 --num_blocks 12 --embed_dim 768 --num_epochs 10 --distill True --teacher_model saved_models/rnng/tokenized_rnng.pt --savepath ptb/test-4gpu-distilled --alpha 0.5 --break_value 5 --trainfile data/tokenized_data/ptb-train.pkl --valfile data/tokenized_data/ptb-val.pkl
```
This is just a test command that trains a small model on a small amount of data on multiple A100 GPUs.

To train the transformer LLM without KD (so it is a purely sequential LLM):
```
torchrun --standalone --nproc_per_node=gpu multi_gpu_gpt2_train.py --ff_dim 768 --num_heads 4 --num_blocks 12 --embed_dim 768 --num_epochs 10 --distill False --savepath ptb/test-4gpu-non_distilled --break_value 5 --trainfile data/tokenized_data/ptb-train.pkl --valfile data/tokenized_data/ptb-val.pkl
```

## Evaluation

We evaluate how these models converge base on perplexity (PPL), which can be seen in the `progress_outputs/` subdirectory.

To evaluate how much of language/syntax these models have learned, we evaluated them using the BLiMP dataset. For reference, see:
Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. Transactions of the Association for Computational Linguistics, 8:377–392.
Github repository: https://github.com/alexwarstadt/blimp/tree/master


## Acknowledgements
Some of this repository code is based on the following repositories:  
- [Unsupervised Recurrent Neural Network Grammars](https://github.com/harvardnlp/urnng)  

## License
MIT