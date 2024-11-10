#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from models import RNNLM, RNNG
from utils import *

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--train_from', default='')
# Model options
parser.add_argument('--w_dim', default=650, type=int, help='hidden dimension for LM')
parser.add_argument('--h_dim', default=650, type=int, help='hidden dimension for LM')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in LM and the stack LSTM (for RNNG)')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
# Optimization options
parser.add_argument('--criterion', default='nll', type=str, choices=['nll', 'cross_entropy'],help='train using nll or cross entropy loss')
parser.add_argument('--count_eos_ppl', default=0, type=int, help='whether to count eos in val PPL')
parser.add_argument('--test', default=0, type=int, help='')
parser.add_argument('--save_path', default='lm.pt', help='where to save the data')
parser.add_argument('--num_epochs', default=18, type=int, help='number of training epochs')
parser.add_argument('--min_epochs', default=8, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--lr', default=0.45, type=float, help='starting learning rate')
parser.add_argument('--decay', default=0.9, type=float, help='')
parser.add_argument('--param_init', default=0.1, type=float, help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=500, help='print stats after this many batches')


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = Dataset(args.train_file)
    val_data = Dataset(args.val_file)  
    vocab_size = int(train_data.vocab_size)   
    
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0), 
           len(val_data)))
    print('Vocab size: %d' % vocab_size)
    
    cuda.set_device(args.gpu)
    if args.train_from == '':
        model = RNNLM(vocab = vocab_size,
                      w_dim = args.w_dim, 
                      h_dim = args.h_dim,
                      dropout = args.dropout,
                      num_layers = args.num_layers)
        if args.param_init > 0:
            for param in model.parameters():    
                param.data.uniform_(-args.param_init, args.param_init)      
    else:
        print('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']
    print("model architecture")
    print(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    model.train()
    model.cuda()
    epoch = 0
    decay= 0
    
    if args.test == 1:
        test_data = Dataset(args.test_file)  
        test_ppl = eval(test_data, model, count_eos_ppl = args.count_eos_ppl)
        sys.exit(0)
        
    # Establish baseline perplexity or cross entropy loss
    best_validation_metrics = eval(val_data, model, count_eos_ppl=args.count_eos_ppl, criterion=args.criterion)
    if args.criterion == 'cross_entropy':
        # This is CE loss
        best_val_metric = best_validation_metrics[0]
        best_val_acc = best_validation_metrics[1]
    else:
        # This is perplexity (PPL)
        best_val_metric = best_validation_metrics
    
    while epoch < args.num_epochs:
        print("Learning rate: ", args.lr)
        print()
        start_time = time.time()
        epoch += 1  
        print('Starting epoch %d' % epoch)
        train_nll = 0
        train_ce_loss = 0
        num_sents = 0
        num_words = 0
        b = 0
        total_correct = 0
        total_seen = 0
        for i in np.random.permutation(len(train_data)):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[i]
            
            if length <= 5:
                continue
                
            # Make the target the last word in the sentence 
            # sentences end with a word(-3), then a period(-2), then the eos token(-1), hence -3
            targets = sents[:, -3]
            targets = targets.cuda()
            sents = sents[:, :-3]
            sents = sents.cuda()
            
            b += 1
            optimizer.zero_grad()
            optimizer.zero_grad()
            
            output = model(sents)
            output_logits = output[0][:, -1, :]
            output_ll = output[1]
            
            if args.criterion == 'nll':
                nll = -(output_ll).mean()
                train_nll += nll.item()*batch_size
                nll.backward()
            else:
                ce_loss = criterion(output_logits, targets)
                train_ce_loss += ce_loss.item()
                ce_loss.backward()
                
                _, pred_idx = torch.max(output_logits, 1)
                correct_pred = (pred_idx == targets)
                total_correct += correct_pred.sum().item()
                total_seen += targets.size(0)
                accuracy = (total_correct / total_seen) * 100
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)   
                
            optimizer.step()
            num_sents += batch_size
            num_words += batch_size * length
            
            if b % args.print_every == 0:
                if args.criterion == 'nll':
                    param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
                    print('Epoch: %d, Batch: %d/%d, LR: %.4f, TrainPPL: %.2f, |Param|: %.4f, BestValPerf: %.2f, Throughput: %.2f examples/sec' % 
                          (epoch, b, len(train_data), args.lr, np.exp(train_nll / num_words), 
                           param_norm, best_val_metric, num_sents / (time.time() - start_time)))
                else:
                    print('Epoch: %d, Batch: %d/%d, LR: %.4f, CE Loss: %.2f' % (epoch, b, len(train_data), args.lr, train_ce_loss))
                    print('Training accuracy for last word prediction: {:.2f}%'.format(accuracy))
                    
        print('--------------------------------')
        print('Checking validation perf...')    
        validation_metrics = eval(val_data, model,  count_eos_ppl=args.count_eos_ppl, criterion=args.criterion)
        if args.criterion == 'cross_entropy':
            # Here val_metric will be cross entropy loss
            val_metric = validation_metrics[0]
            val_acc = validation_metrics[1]
            log_progress(epoch=epoch, criterion=val_metric, acc=val_acc, lr=args.lr)
        else:
            # Here val_metric will be PPL
            val_metric = validation_metrics
            log_progress(epoch=epoch, criterion=val_metric, lr=args.lr)
        print('--------------------------------')
        
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            checkpoint = {
                'args': args.__dict__,
                'model': model.cpu(),
                'word2idx': train_data.word2idx,
                'idx2word': train_data.idx2word
            }
            print('New best validation PPL or Cross Entropy Loss!')
            print('Saving checkpoint to %s' % args.save_path)
            torch.save(checkpoint, args.save_path)
            model.cuda()
        else:
            if epoch > args.min_epochs:
                decay = 1
        if decay == 1:
            args.lr = args.decay*args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            decay = 0
        if args.lr < 0.03:
            break
        print("Finished training")
#         scheduler.step(best_val_metric)

def log_progress(epoch=None, criterion=None, acc=0, lr=None):
    with open("lm_progress_log_2.txt", "a") as f:
        progress = "Epoch: {}, Validation criterion: {}, Accuracy from last batch: {:.2f}%, Learning rate: {}\n".format(epoch, criterion, acc, lr)
        f.write(progress)

def eval(data, model, count_eos_ppl=0, criterion='nll'):
    model.eval()
    num_words = 0
    total_nll = 0
    total_ce_loss = 0
    total_correct = 0
    total_seen = 0
    if criterion == 'cross_entropy':
        ce_loss_func = nn.CrossEntropyLoss()
        
    with torch.no_grad():
        for i in list(reversed(range(len(data)))):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i] 
            if length <= 5: #we ignore length 1 sents in URNNG eval so do this for LM too
                continue
            if args.count_eos_ppl == 1:
                length += 1 
            else:
                # Take out EOS token
                sents = sents[:, :-1] 
                
            targets = sents[:, -2]
            targets = targets.cuda()
            sents = sents[:, :-2]
            sents = sents.cuda()   
            num_words += length * batch_size
            
            output = model(sents)
            
            if criterion == 'nll':
                output_ll = output[1]
                nll = -(output_ll).mean()
                total_nll += nll.item()*batch_size
            else:
                output_logits = output[0][:, -1, :]
                ce_loss = ce_loss_func(output_logits, targets)
                total_ce_loss += ce_loss.item()
                
                _, pred_idx = torch.max(output_logits, 1)
                correct_pred = (pred_idx == targets)
                total_correct += correct_pred.sum().item()
                total_seen += targets.size(0)
                accuracy = (total_correct / total_seen) * 100
    if criterion == 'nll':
        ppl = np.exp(total_nll / num_words)
        print('PPL: %.2f' % (ppl))
        model.train()
        return ppl
    else:
        print('Eval CE Loss: %.2f' % total_ce_loss)
        model.train()
        return total_ce_loss, accuracy

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
