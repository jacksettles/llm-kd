import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gpt2_model import GPT, generate_square_subsequent_mask
from gpt2_data import GPT2Dataset
from rnng_data import RNNGDataset
import os
import sys

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
parser.add_argument("--savepath", default=None, type=str, required=True, help="Where to save the model and progress logs to")
parser.add_argument("--warmup_steps", default=1200, type=int,
                    help="Number of steps to linearly increase the learning rate up to the initial value before beginning decay with cosine annealing scheduler.")
parser.add_argument("--cosine_epochs", default=30, type=int, help="Number of epochs to use a cosine annealing lr scheduler, and then constant lr afterwards.")

class SequentialGPT(pl.LightningModule):
    def __init__(self, model, optimizer_config, args, epoch_length):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.criterion = nn.NLLLoss(reduction='sum')
        self.best_perplexity = float("inf")
        self.args = args
        self.epoch_length = epoch_length

    def forward(self, sents, mask):
#         print(f"\n[GPU]: {torch.distributed.get_rank()} forward pass")
        return self.model(input_ids=sents, attention_mask=mask)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        sents, length, _ = batch
        # Shift inputs and labels for language modeling
        labels = sents[:, 1:]  # Target labels
        sents = sents[:, :-1]  # Input sentences

        # Generate attention mask
        batch_size, seq_length = sents.size()
        mask = generate_square_subsequent_mask(seq_length, device=self.device)

        # Forward pass through the model
        logits = self(sents, mask)

        # Compute NLL loss
        log_probs_word = F.log_softmax(logits, dim=2)
        log_probs_word = log_probs_word.view(batch_size * seq_length, -1)
        labels = labels.reshape(-1)
        nll_loss = self.criterion(log_probs_word, labels)
        total_tokens = int(batch_size * seq_length)
        
        # Log loss and tokens
        self.log("train_nll_loss_sum", nll_loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx="sum")
        self.log("train_total_tokens", total_tokens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, reduce_fx="sum")

        # Free up memory?
        del labels, sents, logits, mask, log_probs_word
        torch.cuda.empty_cache()
        
        # Return the loss for backpropagation
        return nll_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            sents, length, _ = batch
            # Shift inputs and labels
            labels = sents[:, 1:]    # Target labels
            sents = sents[:, :-1]    # Input sentences

            # Generate mask
            batch_size, seq_length = sents.size()
            mask = generate_square_subsequent_mask(seq_length, device=self.device)

            # Forward pass
            logits = self(sents, mask)
            
            # Compute NLL loss
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = log_probs.view(batch_size * seq_length, -1)
            labels = labels.reshape(-1)
            nll_loss = self.criterion(log_probs, labels)
            total_tokens = int(batch_size * seq_length)
            
            # Log loss and tokens
            self.log("val_nll_loss_sum", nll_loss.item(), on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
            self.log("val_total_tokens", total_tokens, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum") 
            
            # Free up memory?
            del labels, sents, logits, mask, log_probs
            torch.cuda.empty_cache()
    
    def _log_progress(self, epoch=None, ppl=None, lr=None, file_name=None):
        # Ensure the directory exists
        directory = "progress_outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{file_name}_progress_log.txt")

        with open(file_path, "a") as f:
            progress = f"Epoch: {epoch+1}, Validation PPL: {ppl}, Learning rate: {lr}\n"
            f.write(progress)
            
    def _save_snapshot(self):
        checkpoint = {
            "MODEL":self.model,
            "ARGS":self.args.__dict__,
            "EPOCH":self.current_epoch
        }
        save_path = self.args.savepath
        save_file = "saved_models/" + save_path + ".pt"
        print(f"New lowest perplexity, saving model to {save_file}")
        torch.save(checkpoint, save_file)

    def on_validation_epoch_end(self):
        print("Finished eval!!")
        total_nll_loss = self.trainer.logged_metrics["val_nll_loss_sum"]
        total_tokens = self.trainer.logged_metrics["val_total_tokens"]

        # Compute average NLL loss and perplexity
        avg_val_nll_loss = total_nll_loss / total_tokens
        val_ppl = torch.exp(avg_val_nll_loss)

        lr_schedulers = self.lr_schedulers()
        current_lr = lr_schedulers.get_last_lr()[0]
        if torch.distributed.get_rank() == 0:
            self._log_progress(epoch=self.current_epoch, ppl=val_ppl, lr=current_lr, file_name=self.args.savepath)
            if val_ppl < self.best_perplexity:
                self.best_perplexity = val_ppl
                self._save_snapshot()
                    
    def make_scheduler(self, optimizer, args, epoch_length):
        def lr_lambda(current_step: int, args):
            warmup_steps = args.warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, args))
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_length*10, T_mult=2, eta_min=1e-7)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                    schedulers=[warmup_scheduler, cosine_scheduler],
                                                    milestones=[args.warmup_steps])
        return scheduler

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)

        # Define scheduler
        self.scheduler = self.make_scheduler(self.optimizer, self.args, self.epoch_length)

        # Return optimizer and scheduler
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",  # Step the scheduler after every step (or "epoch" for per-epoch stepping)
                "frequency": 1,      # Apply the scheduler every 1 step/epoch
            },
        }
    

class GPT2DataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size=1, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data = train_data
        self.val_data = val_data
        self.train_sampler = None
        self.val_sampler = None

    def setup(self, stage=None):
        # Set up samplers if distributed training is initialized
        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(self.train_data)
            self.val_sampler = DistributedSampler(self.val_data, shuffle=False)

    def tensor_collate_fn(self, batch):
        # Extract the elements from the batch
        sents = batch[0][0]  # Already a tensor of shape (batch_size, sequence_length)
        length = batch[0][1]  # Scalar
        gold_binary_actions = batch[0][2]  # List of lists

        # Convert the list of lists to a tensor
        gold_binary_actions_tensor = torch.tensor(gold_binary_actions)
        append_values = torch.tensor([0, 1]).unsqueeze(0).repeat(gold_binary_actions_tensor.size(0), 1)  # Shape: (num_rows, 2)
#         print(gold_binary_actions_tensor.shape)

        # Concatenate the append_values to the actions tensor along dimension 1 (columns)
        actions = torch.cat((gold_binary_actions_tensor, append_values), dim=1)
#         print(actions.shape)
        
        return sents, length, actions
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),  # Shuffle only if no sampler
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            persistent_workers=False,
            collate_fn=self.tensor_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            persistent_workers=False,
            collate_fn=self.tensor_collate_fn
        )
    
def organize_data(args):
    print("Loading datasets")
    train_set = RNNGDataset(args.trainfile)
    val_set = RNNGDataset(args.valfile)
    print("Loaded training and validation data")
    vocab_size = int(train_set.vocab_size)
    print(f"Vocab size: {vocab_size}")

    # Filter and create data lists
    print("Filtering batches")
    train_list = [(train_set[i][0], train_set[i][1], train_set[i][5]) 
              for i in range(len(train_set)) 
              if (train_set[i][0].size(0) == 64) and (train_set[i][1] > 1)]
    del train_set
    print("Train list created")
    val_list = [(val_set[i][0], val_set[i][1], val_set[i][5]) 
              for i in range(len(val_set)) 
              if (val_set[i][0].size(0) == 64) and (val_set[i][1] > 1)]
    del val_set
    print("Validation list created")
    epoch_length = len(train_list)

    # Convert to GPT2Dataset
    train_data = GPT2Dataset(train_list)
    del train_list
    val_data = GPT2Dataset(val_list)
    del val_list
    print("Datasets prepared")
    return train_data, val_data, vocab_size, epoch_length
    
def main(args):
    print("Organizing data")
    train_data, val_data, vocab_size, epoch_length = organize_data(args)
    args.warmup_steps = int(epoch_length * 0.05)
    
    print("Making the model")
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
    
    print("Wrapping the model in the PL module")
    model = SequentialGPT(model=model, optimizer_config={"lr": args.lr}, args=args, epoch_length=epoch_length)

    if args.distill == 0:
        log_name = "non_distilled"
    elif args.distill == 1:
        log_name = "distilled"
    print("Making logger")
    logger = CSVLogger(
        save_dir="logs",           # Directory to save the logs
        name=log_name,             # Experiment name (creates a subfolder)
        version="batch_size_64"               # Version number (creates another subfolder)
    )
    
    print("Making the trainer object")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    log_every = epoch_length // 4
    trainer = Trainer(
        strategy="ddp",
        logger=logger,
        deterministic=True,
        devices=num_gpus,
        accelerator="gpu",
        max_epochs=args.num_epochs,
        precision=16,
        enable_checkpointing=False
    )
    print("Intializing the data module")
    data_module = GPT2DataModule(train_data, val_data, batch_size=1, num_workers=8)
    print("Fitting the trainer")
    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    main(args)

