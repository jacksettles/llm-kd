import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
import argparse
import time
import os
import sys
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gpt2_model import GPT, generate_square_subsequent_mask

# Using same dataset class as URNNG (Kim et al., 2019), 
# so when we do knowledge distillation, data is already prepared for the RNNG model
from data import Dataset
from tokenized_models import RNNG


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--trainfile", default="data/tokenized_data/ptb-train.pkl", type=str, help="path to train data file")
parser.add_argument("--valfile", default="data/tokenized_data/ptb-val.pkl", type=str, help="path to validation data file")
parser.add_argument("--testfile", default="data/tokenized_data/ptb-test.pkl", type=str, help="path to test data file")
parser.add_argument("--embed_dim", default=768, type=int, help="Dimension size of the embedding layer")
parser.add_argument("--max_length", default=1024, type=int, help="Max sequence length")
parser.add_argument("--embed_dropout", default=0.1, type=float, help="Probability of dropout for the embedding layer")
parser.add_argument("--num_blocks", default=6, type=int, help="Number of transformer blocks")
parser.add_argument("--num_heads", default=8, type=int, help="Number of attention heads")
parser.add_argument("--ff_dim", default=2048, type=int, help="Dimension of the feedforward layer in each transformer block")
parser.add_argument("--attn_dropout", default=0.1, type=float, help="Probability of dropout for the attention modules")
parser.add_argument("--ff_dropout", default=0.1, type=float, help="Probability of dropout for the feedforward layers")
parser.add_argument("--tokenizer_vocab", default="tokenizers/rnng/vocab.json", type=str, help="Path to tokenizer vocab.json file")
parser.add_argument("--tokenizer_merges", default="tokenizers/rnng/merges.txt", type=str, help="Path to tokenizer merges.txt file")
parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate for the model's optimizer")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of epochs to train for")
parser.add_argument("--min_epochs", default=25, type=int, help="Minimum number of epochs to train for before tracking # of bad epochs")
parser.add_argument("--print_every", default=1000, type=int, help="How often you want to print training updates")
parser.add_argument("--count_eos_ppl", default=1, type=int, help="Whether to count perplexity for eos token during eval")
parser.add_argument("--break_value", default=3, type=int,
                    help="When PPL is worse than before, and past min_epochs, use this threshold of bad epochs. So if PPL is worse than the best PPL this many times, break.")
parser.add_argument("--distill", default=0, type=int, help="Whether or not you want to train this GPT2 model using knowledge distillation")
parser.add_argument("--teacher_model", default=None, type=str, help="The path to the trained model (RNNG) for knowledge distillation")
parser.add_argument("--temperature", default=2.0, type=float, help="If doing KD, scale the output logits by this temperature variable")
parser.add_argument("--alpha", default=0.5, type=float, help="Weight applied to combined loss during KD. alpha*NLLLoss + (1-alpha)*KLDiv")
parser.add_argument("--savepath", default=None, type=str, required=True, help="Where to save the models to")
parser.add_argument("--warmup_steps", default=2000, type=int,
                    help="Number of steps to linearly increase the learning rate up to the initial value before beginning decay with cosine annealing scheduler.")
parser.add_argument("--cosine_epochs", default=30, type=int, help="Number of epochs to use a cosine annealing lr scheduler, and then constant lr afterwards.")

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def eval(val_data=None, model=None, device="cpu", world_size=1, rank=None):
    model.eval()
    print("Running eval!!!!!!!!")
    val_vocab_size = int(val_data.vocab_size)
    criterion = nn.NLLLoss(reduction='sum')
    total_tokens = 0
    total_sents = 0
    val_nll_loss = 0
    
    split_indices = np.array_split(np.random.permutation(len(val_data)), world_size)
    min_length = min(len(split) for split in split_indices)
    truncated_splits = [split[:min_length] for split in split_indices]
    indices_for_rank = truncated_splits[rank]
    
    dist.barrier()
    with torch.no_grad():
        for i in indices_for_rank:
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = val_data[i]
            if length == 1: # length 1 sents are ignored
                continue

            sents = sents.to(device)
            labels = sents[:, 1:]
            sents = sents[:, :-1]
            batch_size, length = sents.size(0), sents.size(1)
                      
            mask = generate_square_subsequent_mask(length, device=device)

#             dist.barrier()
            logits = model(input_ids=sents, attention_mask=mask)
            #torch.cuda.synchronize() # sync
#             print(f"Rank {rank} finished EVAL forward pass for batch {i}")

            log_probs_word = F.log_softmax(logits, 2)

            log_probs_word = log_probs_word.view(batch_size*length, -1)
            labels = labels.reshape(-1)
            
#             dist.barrier()
            nll_loss = criterion(log_probs_word, labels)
            #torch.cuda.synchronize() # sync
#             print(f"Rank {rank} finished EVAL loss calculation for batch {i}")

            val_nll_loss += nll_loss.item()
            
            total_tokens += batch_size*length
            total_sents += batch_size
            dist.barrier()
            
        # Aggregate validation loss across all processes
        val_nll_loss_tensor = torch.tensor(val_nll_loss, device=device)
        total_tokens_tensor = torch.tensor(total_tokens, device=device)

        dist.barrier()
        # Perform all-reduce to sum up the validation loss and tokens across all processes
        #torch.cuda.synchronize() # sync
        print(f"Rank {dist.get_rank()}: Preparing to run dist.all_reduce on val_nll_loss_tensor")
        dist.all_reduce(val_nll_loss_tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {dist.get_rank()}: Completed dist.all_reduce on val_nll_loss_tensor\n")

#         dist.barrier()
        #torch.cuda.synchronize() # sync
        print(f"Rank {dist.get_rank()}: Preparing to run dist.all_reduce on total_tokens_tensor")
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {dist.get_rank()}: Completed dist.all_reduce on total_tokens_tensor\n")
        #dist.barrier()

        # Compute the global average validation loss
        avg_val_nll_loss = val_nll_loss_tensor.item() / total_tokens_tensor.item()

        # Calculate perplexity
        val_ppl = torch.exp(torch.tensor(avg_val_nll_loss))
        model.train()
#         dist.barrier()
        # Only log from rank 0
        if dist.get_rank() == 0:
            return avg_val_nll_loss, val_ppl.item()
        else:
            return None, None  # Only return results for rank 0
    
def lr_lambda(current_step: int, args):
    warmup_steps = args.warmup_steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0
    
def main(rank, world_size, args):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    print("Loading in training data!!")
    train_data = Dataset(args.trainfile)
    print("Training data loaded successfully")
    print()
    print("Loding in validation data!!")
    val_data = Dataset(args.valfile)
    print("Validation data loaded successfully")
    print()
    vocab_size = int(train_data.vocab_size)
    print(f"Rank {rank}: Vocab size: {vocab_size}")
    
    # Define model parameters
    embed_dim = args.embed_dim
    max_len = args.max_length 
    embed_dropout = args.embed_dropout
    num_blocks = args.num_blocks 
    num_heads = args.num_heads 
    ff_dim = args.ff_dim
    attn_dropout = args.attn_dropout
    ff_dropout = args.ff_dropout
    save_path = args.savepath

    # Initialize GPT model
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_len=max_len,
        embed_dropout=embed_dropout,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ff_dim=ff_dim,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout
    )
    model.to(device)
    
    # Wrap the model in DistributedDataParallel
    model = DDP(model, device_ids=[rank])#, find_unused_parameters=True)
    
    # If you are doing knowledge distillation for training GPT2
    # Teacher as of now is a trained RNNG
    if args.distill == 1:
        temp = args.temperature
        alpha = args.alpha
        print(f"Rank {rank}: Loading teacher model from {args.teacher_model}")
        checkpoint = torch.load(args.teacher_model, map_location=device)
        rnng = checkpoint['model']
        rnng.to(device)
        rnng = DDP(rnng, device_ids=[rank])
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    # Tokenizer to decode the outputs back to tokens
    # The datasets have already been tokenized, and the urnng.data Dataset 
    # class already returns sents that are tensors of token indices, so no need to tokenize again
#     tokenizer = ByteLevelBPETokenizer(args.tokenizer_vocab, args.tokenizer_merges)
        
    lr = args.lr
    num_epochs = args.num_epochs
    warmup_steps = args.warmup_steps
    epoch_length = len(train_data)
#     cosine_steps = epoch_length*num_epochs#args.cosine_epochs
#     T_max = cosine_steps - warmup_steps
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, args))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_length*10, T_mult=2, eta_min=1e-9)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                schedulers=[warmup_scheduler, cosine_scheduler],
                                                milestones=[warmup_steps])

    criterion = nn.NLLLoss(reduction='sum')
    
    # This variable tracks the number of times PPL has failed to improve (decrease)
    # If we hit a certain number of "bad epochs", cut training short
    num_bad_epochs = 0
    print("Running first evaluation on validaiton data to get baseline!")
    dist.barrier()
    initial_val_nll_loss, init_val_ppl = eval(val_data=val_data, model=model, device=device, world_size=world_size, rank=rank)
    print("Eval is done!")
    print()
    best_loss = initial_val_nll_loss
    best_ppl = init_val_ppl
    
    print("------------")
    print("Initial PPL: ", best_ppl)
    print("Start training")
    print("------------")

    # Training loops
    for epoch in range(num_epochs):
#         dist.barrier()
        model.train()
        total_loss = 0.0
        total_tokens = 0
        split_indices = np.array_split(np.random.permutation(len(train_data)), world_size)
        min_length = min(len(split) for split in split_indices)
        truncated_splits = [split[:min_length] for split in split_indices]
        indices_for_rank = truncated_splits[rank]
        dist.barrier()

        print(f"Length of indices for rank {rank}: {len(indices_for_rank)}")
        print()

        for i in indices_for_rank:
            dist.barrier()
            optimizer.zero_grad()
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[i]
            print(f"Rank {rank} - sentence length {length}")
            print(f"Rank {rank} processing batch of length {len(sents)}")
            print()
            if length == 1:
                continue
            
            # Move sents to GPU
            sents = sents.to(device)
            
#             dist.barrier()
            if args.distill == 1:
                actions = gold_binary_trees
                # Don't need gradients for the teacher model
                with torch.no_grad():
                    teacher_output = rnng.forward(sents, actions)
                teacher_word_logits, _, _ = teacher_output
                teacher_word_probs = F.softmax(teacher_word_logits / temp, dim=2)
                
            labels = sents[:, 1:]
            sents = sents[:, :-1]
            batch_size, length = sents.size(0), sents.size(1)

            mask = generate_square_subsequent_mask(length, device=device)
            
#             dist.barrier()
            logits = model(input_ids=sents, attention_mask=mask)
            #torch.cuda.synchronize() # sync
#             print(f"Rank {rank} finished forward pass for batch {i}")
            
            if args.distill == 1:
                # Apply temperature scaling if doing KD
                # and get Kullback-Leibler divergence
                temp_scaled_log_probs_word = F.log_softmax(logits / temp, dim=2)
                kl_loss = criterion_kl(temp_scaled_log_probs_word, teacher_word_probs)
                #torch.cuda.synchronize() # sync

            # Separate variable for non-temp-scaled log probs to calculate NLL Loss
            log_probs_word = F.log_softmax(logits, dim=2)
            #torch.cuda.synchronize() # sync

            log_probs_word = log_probs_word.view(batch_size*length, -1)
            labels = labels.reshape(-1)
            
#             dist.barrier()
            nll_loss = criterion(log_probs_word, labels)
            #torch.cuda.synchronize() # sync
            print(f"Rank {rank} calculated loss for batch {i}")

#             dist.barrier()
            print(f"ZH. {rank} hits barrier here ...")
            if args.distill == 1:
                combined_loss = (alpha*nll_loss) + ((1-alpha)*kl_loss)
                combined_loss.backward()
                #torch.cuda.synchronize() # sync
            else:
                nll_loss.backward()
                #torch.cuda.synchronize() # sync
            print(f"Rank {rank} finished backward pass for batch {i}")
            
            total_loss += nll_loss.item()
            total_tokens += batch_size*length
            
#             dist.barrier()
            optimizer.step()
            #torch.cuda.synchronize() # sync
            
#             dist.barrier()
            scheduler.step()
            #torch.cuda.synchronize() # sync
            print(f"Rank {rank} finished optimizer step for batch {i}")
#             dist.barrier()
            
        # End of epoch; eval here
        print("End of epoch!")
        print(f"Rank {rank} reached end-of-epoch barrier")
        dist.barrier() #insure all processes reach the same point before proceeding
        print(f"Rank {rank} made it through end-of-epoch barrier")
        avg_val_nll_loss, val_ppl = eval(val_data=val_data, model=model, device=device, world_size=world_size, rank=rank)
#         dist.barrier()
        print("Val ppl: ", val_ppl)
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Avg. Validation Loss: {avg_val_nll_loss}, Val PPL: {val_ppl}')
            log_progress(epoch=(epoch+1), ppl=val_ppl, lr=current_lr, file_name=save_path)
#         dist.barrier()
        if rank == 0:
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                checkpoint = {
                    'args': args.__dict__,
                    'model': model.cpu(),
                    'word2idx': train_data.word2idx,
                    'idx2word': train_data.idx2word
                }
                save_file = "saved_models/" + save_path + ".pt"
                print(f"New lowest perplexity, saving model to {save_file}")
                torch.save(checkpoint, save_file)
                model.to(device)
            else:
                if epoch > args.min_epochs:
                    num_bad_epochs += 1
                if num_bad_epochs == args.break_value:
                    print(f"Breaking after {epoch} epochs because PPL failed to improve")
                    break
#         dist.barrier()
    cleanup()

def log_progress(epoch=None, ppl=None, lr=None, file_name=None):
    # Ensure the directory exists
    directory = "progress_outputs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{file_name}_progress_log.txt")
    
    with open(file_path, "a") as f:
        progress = f"Epoch: {epoch}, Validation PPL: {ppl}, Learning rate: {lr}\n"
        f.write(progress)
            
if __name__ == '__main__':
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    world_size = torch.cuda.device_count()
    print("World size", world_size)
    
    start_time = time.perf_counter()
#     main(rank=0, world_size=1, args=args)
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total training time (min:sec): {minutes}:{seconds}")
