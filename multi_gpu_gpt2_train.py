import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from rnng_data import RNNGDataset
from gpt2_data import GPT2Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# from transformers import GPT2Tokenizer
import argparse
import time
import os
import sys
import numpy as np

from gpt2_model import GPT, generate_square_subsequent_mask
from tokenized_models import RNNG

parser = argparse.ArgumentParser(description='Multi-GPU language model training')

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


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        snapshot_path: str,
        distill: int,
        min_epochs: int,
        break_value: int
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.distill = distill
        if self.distill == 1:
            self.temp = args.temperature
            self.alpha = args.alpha
            print(f"Rank {self.gpu_id}: Loading teacher model from {args.teacher_model}")
            checkpoint = torch.load(args.teacher_model, map_location=self.device)
            rnng = checkpoint['model'].to(self.device)
            self.teacher_model = DDP(rnng, device_ids=[self.gpu_id])
            self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
            
        if os.path.exists(f"saved_models/{snapshot_path}"):
            print(f"Rank {self.gpu_id} loading snapshot")
            self._load_snapshot(f"saved_models/{snapshot_path}")

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.criterion = nn.NLLLoss(reduction='sum')
        self.min_epochs = min_epochs
        self.break_value = break_value

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model = snapshot["MODEL"].to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, sents, length, gold_binary_trees):
        self.optimizer.zero_grad()
        
        if self.distill == 1:
            actions = gold_binary_trees
            # Don't need gradients for the teacher model
            with torch.no_grad():
                teacher_output = self.teacher_model.forward(sents, actions, device=self.device)
            teacher_word_logits, _, _ = teacher_output
            teacher_word_probs = F.softmax(teacher_word_logits / self.temp, dim=2)
        else:
            _ = gold_binary_trees
        
        labels = sents[:, 1:]
        sents = sents[:, :-1]
        
        batch_size, length = sents.size(0), sents.size(1)
        mask = generate_square_subsequent_mask(length, device=self.device)
        logits = self.model(input_ids=sents, attention_mask=mask)
        
        if self.distill == 1:
            temp_scaled_log_probs_word = F.log_softmax(logits / self.temp, dim=2)
            kl_loss = self.criterion_kl(temp_scaled_log_probs_word, teacher_word_probs)
        
        log_probs_word = F.log_softmax(logits, dim=2)
        log_probs_word = log_probs_word.view(batch_size*length, -1)
        labels = labels.reshape(-1)
        nll_loss = self.criterion(log_probs_word, labels)
        if self.distill == 1:
            combined_loss = (self.alpha*nll_loss) + ((1-self.alpha)*kl_loss)
            combined_loss.backward()
        else:
            nll_loss.backward()
#         torch.cuda.synchronize()

        self.optimizer.step()
        self.scheduler.step()
#         dist.barrier()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for sents, length, gold_binary_trees in self.train_data:
#             if length == 1:
#                 continue
            # sents has to be squeezed to remove the singleton batch dimension made by the dataloader
#             sents = sents.squeeze(0).to(self.gpu_id)
            sents = sents.to(self.gpu_id)
            try:
                self._run_batch(sents, length, gold_binary_trees)
            except Exception as e:
                print(repr(e))
            
    def _run_eval(self):
        self.model.eval()
        print(f"Rank {self.gpu_id} running eval!")
        val_nll_loss = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0, device=self.device)
        total_sents = torch.tensor(0, device=self.device)
        with torch.no_grad():
            for sents, length, _ in self.val_data:
                sents = sents.to(self.gpu_id) 
                labels = sents[:, 1:]
                sents = sents[:, :-1]
                batch_size, length = sents.size(0), sents.size(1)
                mask = generate_square_subsequent_mask(length, device=self.device)
                logits = self.model(input_ids=sents, attention_mask=mask)
                log_probs_word = F.log_softmax(logits, 2)
                log_probs_word = log_probs_word.view(batch_size*length, -1)
                labels = labels.reshape(-1)
                nll_loss = self.criterion(log_probs_word, labels)
                val_nll_loss += nll_loss
                n_tokens = torch.tensor(batch_size*length, device=self.device)
                total_tokens += n_tokens
                total_sents += torch.tensor(batch_size, device=self.device)
#                 dist.barrier()
            
            print(f"Rank {self.gpu_id} done with eval")
            dist.barrier()
            dist.all_reduce(val_nll_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            avg_val_nll_loss = val_nll_loss / total_tokens
            val_ppl = torch.exp(avg_val_nll_loss)

        if dist.get_rank() == 0:
            return avg_val_nll_loss.item(), val_ppl.item()
        else:
            return None, None
        
        
    def _save_snapshot(self, epoch, args):
        snapshot = {
            "ARGS": args.__dict__,
            "MODEL": self.model.module,
            "EPOCHS_RUN": epoch,
        }
        save_file = "saved_models/" + self.snapshot_path
        torch.save(snapshot, save_file)
        print(f"Epoch {epoch} | Training snapshot saved at {save_file}")

        
    def _log_progress(self, epoch=None, ppl=None, lr=None, file_name=None):
        # Ensure the directory exists
        directory = "progress_outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{file_name}_progress_log.txt")

        with open(file_path, "a") as f:
            progress = f"Epoch: {epoch}, Validation PPL: {ppl}, Learning rate: {lr}\n"
            f.write(progress)
        
        
    def train(self, max_epochs: int, args):
        print("Running initial eval to get baseline metrics")
        initial_val_nll_loss, init_val_ppl = self._run_eval()
        best_loss = initial_val_nll_loss
        best_ppl = init_val_ppl
        num_bad_epochs = 0
        print("Training!\n")

        for epoch in range(self.epochs_run, max_epochs):
            dist.barrier()
            self.model.train()
            self._run_epoch(epoch)
            
            try:
                avg_val_nll_loss, val_ppl = self._run_eval()
            except Exception as e:
                print(repr(e))
            current_lr = self.scheduler.get_last_lr()[0]
            if self.gpu_id == 0:
                print(f'Epoch: {epoch+1}/{max_epochs}, Avg. Validation Loss: {avg_val_nll_loss}, Val PPL: {val_ppl}')
                save_path = self.snapshot_path.split('.')[0]
                self._log_progress(epoch=(epoch+1), ppl=val_ppl, lr=current_lr, file_name=save_path)
           
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    self._save_snapshot(epoch, args)
                else:
                    if epoch > self.min_epochs:
                        num_bad_epochs += 1
                    if num_bad_epochs == self.break_value:
                        print(f"Breaking after {epoch} epochs because PPL failed to improve")
                        break
        dist.barrier()


def lr_lambda(current_step: int, args):
    warmup_steps = args.warmup_steps
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

                
def make_scheduler(optimizer, args, epoch_length):
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, args))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_length*10, T_mult=2, eta_min=1e-9)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                schedulers=[warmup_scheduler, cosine_scheduler],
                                                milestones=[int(epoch_length*0.05)])
    return scheduler


def load_train_objs(args):
    train_set = RNNGDataset(args.trainfile)  # load your dataset
    val_set = RNNGDataset(args.valfile)
    print("Loaded training and validation data")
    vocab_size = int(train_set.vocab_size)
    print(f"Vocab size: {vocab_size}")
    
    print("Making data lists")
    # Basically only keep the batches that have a size of 16, leftover batches of smaller szes throw devices out of sync
    train_list = [(train_set[i][0], train_set[i][1], train_set[i][5]) 
                for i in range(len(train_set)) 
                if (train_set[i][0].size(0) == 16) and (train_set[i][1] > 1)]
    num_batches = len(train_list)
    del train_set
    print("Train list made, train set deleted")
    val_list = [(val_set[i][0], val_set[i][1], val_set[i][5]) 
              for i in range(len(val_set)) 
              if (val_set[i][0].size(0) == 16) and (val_set[i][1] > 1)]
    del val_set
    print("Val list made, Val set deleted")
    
    train_data = GPT2Dataset(train_list)
    del train_list
    print("Train data made, train list deleted")
    val_data = GPT2Dataset(val_list)
    del val_list
    print("Val data made, val list deleted")

    
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        max_len=args.max_length,
        embed_dropout=args.embed_dropout,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args, num_batches)
    return train_data, val_data, model, optimizer, scheduler

def tensor_collate_fn(batch):
    # Extract the elements from the batch
    sents = batch[0][0]  # Already a tensor of shape (batch_size, sequence_length)
    length = batch[0][1]  # Scalar
    gold_binary_actions = batch[0][2]  # List of lists

    # Convert the list of lists to a tensor
    gold_binary_actions_tensor = torch.tensor(gold_binary_actions)
    append_values = torch.tensor([0, 1]).unsqueeze(0).repeat(gold_binary_actions_tensor.size(0), 1)  # Shape: (num_rows, 2)

    # Concatenate the append_values to the actions tensor along dimension 1 (columns)
    actions = torch.cat((gold_binary_actions_tensor, append_values), dim=1)

    return sents, length, actions

def prepare_dataloader(dataset: Dataset, batch_size: int=1):
    return DataLoader(
        dataset,
        batch_size=batch_size, #RNNGDataset is already pre-batched, so batch_size=1 will return 1 batch of 16 sentences
        pin_memory=True,
        shuffle=False,
        collate_fn=tensor_collate_fn,
        num_workers=0,
        sampler=DistributedSampler(dataset)
    )

    return dataloader


def main(args, total_epochs: int):
    ddp_setup()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Loading training objects!")
    train_dataset, val_dataset, model, optimizer, scheduler = load_train_objs(args)
    
    print("Making DataLoaders!")
    train_data = prepare_dataloader(train_dataset, batch_size=1)
    val_data = prepare_dataloader(val_dataset, batch_size=1)
    
    snapshot_path = args.savepath + ".pt"
    
    print("Making Trainer object...")
    # print(f"This is the args distill: {args.distill}")
    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, snapshot_path, args.distill, args.min_epochs, args.break_value)
    trainer.train(total_epochs, args)
    destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()

    start_time = time.perf_counter()
    main(args, args.num_epochs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total training time (min:sec): {minutes}:{seconds}")