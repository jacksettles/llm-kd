#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join("..")))

# from data import Dataset


# In[2]:


class InputEmbedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        """
        Initialize the InputEmbedding module.
        
        Args:
            embed_dim (int): The dimensionality of the input embedding
            vocab_size (int): The size of the vocabulary
        """
        super().__init__()
        # Store the dimensionality and vocabulary size
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Create embedding layer to map vocabulary to an embed_dim-dimensionalspace
        # The embedding layer should have shape (vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        """
        Perform the forward ass of the InputEmbedding module.
        
        Args:
            x (tensor): The input tensor.
            
        Returns:
            tensor: The embedded input tensor after scaling it by the square root of the dimensionality.
        """
        # Embed the input tensor using the embedding layer
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded_input = self.embedding(x)
        scaled_embedded_input = embedded_input * torch.sqrt(torch.tensor(self.embed_dim))
        return scaled_embedded_input


# In[3]:


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int = 512, max_seq_len: int = 100, dropout: float = 0.1,):
        """Initialize the PositionalEncoding module."""
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        # Precompute the positional encoding matrix
        self.positional_encoding = self._precompute_positional_encoding(max_seq_len, embed_dim)
        
    def _precompute_positional_encoding(self, max_seq_len, embed_dim):
        """Precompute the positional encoding matrix."""
        with torch.no_grad():
            # Create a positional encoding matrix of shape (max_seq_len,embed_dim)
            positional_encoding = torch.zeros(max_seq_len, embed_dim)
            # Create a tensor 'pos' with values [0, 1, 2, ..., max_seq_len - 1] (max_seq_len, 1)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            # Compute the positional encoding matrix
            division_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
            positional_encoding[:, 0::2] = torch.sin(position * division_term)
            positional_encoding[:, 1::2] = torch.cos(position * division_term)
            # Shape (max_seq_len, embed_dim) -> (1, max_seq_len, embed_dim)
            positional_encoding = positional_encoding.unsqueeze(0)
        
        return positional_encoding
    
    def forward(self, x):
        """Perform the forward pass of the PositionalEncoding module."""
        # Add the positional encoding matrix to the input tensor
        x = x + self.positional_encoding[:, : x.size(1)].to(x.device)
        # Apply dropout to the input tensor
        x = self.dropout(x)
        return x


# In[4]:


class LayerNormalization(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        """Initialize the LayerNormalization module."""
        super().__init__()
        self.eps = eps
        # Create two learnable parameters to scale and shift the normalized input
        self.gain = nn.Parameter(torch.Tensor(embed_dim).uniform_())
        self.bias = nn.Parameter(torch.Tensor(embed_dim).normal_())
        
    def forward(self, x):
        """Perform the forward pass of the LayerNormalization module."""
        # Compute the mean and standard deviation of the input tensor
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Zero center by subtracting the mean from the input tensor
        return (x - mean) / (std + self.eps) * self.gain + self.bias


# In[5]:

def geglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, intermediate_size: int, dropout: float = 0.1):
        """Initialize the FeedForwardBlock module."""
        # Update on 7/9/2024 -
        # Intermediate size of fc1 was doubled to accomadate new GeGLU activation function
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size*2, bias=False)
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Since you apply this layer normalization after fc1 and before fc2,
        # embed_dim is same as intermediate_size instead of original embed_dim
        self.layer_norm = LayerNormalization(embed_dim=intermediate_size) 
        
    # Update on 7/9/2024 -
    # Added a layer_norm to follow in line with NormFormer (Shleifer and Ott, 2022)
    # Also added GeGLU instead of GELU to match LTG-BERT
    def forward(self, x):
        x_intermediate = geglu(self.fc1(x))
        x_intermediate = self.dropout(self.layer_norm(x_intermediate))
        x_output = self.fc2(x_intermediate)
        return x_output


# In[6]:


def generate_square_subsequent_mask(size: int, device: torch.device = "cpu"):
    """Generate a square mask for the sequence."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=0)
    # Turn boolean mask into float mask
    mask = mask.long()
    return mask.unsqueeze(0) # Add batch dimension


# In[7]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 512, num_heads: int = 8, attn_dropout: float = 0.1, ff_dropout: float = 0.1, max_len=100):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % self.num_heads == 0, "Invalid heads and embed_dim configuration; embed_dim must be evenly divisble by num_heads (there should be no remainder)"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(ff_dropout)
        self.layer_norm = LayerNormalization(embed_dim=embed_dim)
        # Create a buffer to store the mask wth no gradient
        # Shape: (1, max_len, max_len)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )
        
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.size()
        # Apply linear transformations to the input tensor
        # Take input tensor and apply linear transformations
        # then split the tensor into num_heads and head_dim
        # transpose the tensor into the correct order
        q = self.query(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        attention = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(q.size(-1))
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
            
        attention = self.attn_dropout(F.softmax(attention, dim=-1))

        y = torch.einsum('bhij,bhjd->bhid', attention, v)
        
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply linear transformation and dropout
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        return self.proj_dropout(self.layer_norm(self.proj(y)))


# In[8]:


class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout: float = 0.1):
        """Initialize the ResidualConnection module."""
        super().__init__()
        self.layer_norm = LayerNormalization(embed_dim=embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Perform the forward pass of the ResidualConnection module."""
        # Apply layer normalization
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        normalized_x = self.layer_norm(x)
        
        # Apply sublayer (e.g. feedforward block)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        sublayer_output = sublayer(normalized_x)
        
        # Add residual connection and apply dropout
        # (batch_size, seq_len, embed_dim) + (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        residual_output = x + self.dropout(sublayer_output)
        return residual_output


# In[9]:


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        """Initialize the ProjectionHead module."""
        super().__init__()
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        """Perform forward pass of the ProjectionHead module."""
        # Apply linear transformation to the input tensor
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        return self.fc(x)


# In[10]:


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        # Initialize the multi-head self-attention mechanism
        self.MultiHeadAttention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            max_len=max_len,
        )
        # Initialize the feed-forward block
        self.feed_forward = FeedForwardBlock(
            embed_dim=embed_dim,
            intermediate_size=ff_dim,
            dropout=ff_dropout,
        )
        # Initialize residual connections
        self.residual_connection1 = ResidualConnection(embed_dim=embed_dim, dropout=dropout)
        self.residual_connection2 = ResidualConnection(embed_dim=embed_dim, dropout=dropout)
        
    def forward(self, x, attention_mask=None):
        # Apply self-attention mechanism with residual connection
        x_with_attention = self.residual_connection1(x, lambda x: self.MultiHeadAttention(x, mask=attention_mask))
        # Apply feed-forward block with residual connection
        x_with_ff = self.residual_connection2(x_with_attention, self.feed_forward)
        return x_with_ff


# In[11]:


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        max_len: int = 512,
        embed_dropout: float = 0.1,
        num_blocks: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = InputEmbedding(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )
        self.positional_embedding = PositionalEncoding(
            embed_dim=embed_dim,
            max_seq_len=max_len,
            dropout=embed_dropout,
        )
        self.blocks = nn.ModuleList([DecoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            max_len=max_len,
        ) for _ in range(num_blocks)])
        
        self.projection_head = ProjectionHead(embed_dim=embed_dim, vocab_size=vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # get_embeddings() method takes care of most of the processing
        x = self.get_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        
        # Linear layer for output logits
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        x = self.projection_head(x)
        
        return x
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Shape: (batch_size, seq_len) - > (seq_len)
        seq_len = input_ids.size(1)
        assert seq_len <= self.max_len, "Sequence is longer than model capacity"
        
        # Token embedding
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.token_embedding(input_ids)
        
        # Add positional embedding
        x = self.positional_embedding(x)
        
        # Forward through decoder blocks
        # Output of each block iis the hidden state of the transformer
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
            
        return x

    def get_emb_no_context(self, input_ids: torch.Tensor):
        return self.token_embedding(input_ids)


class GPTDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.end_token = tokenizer.eos_token_id #token_to_id("</s>")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        input_txt = self.tokenizer(text, truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
        text_len = input_txt.size(0)
        if text_len < self.max_length:
            padding_len = self.max_length - text_len
            padding = torch.tensor([self.end_token] * padding_len)
            input_ids = torch.cat((input_txt, padding), dim=0)
            label = torch.cat((input_txt[1:], torch.tensor([self.end_token]), padding), dim=0)
        else:
            input_ids = input_txt[:self.max_length]
            label = torch.cat((input_txt[1:self.max_length], torch.tensor([self.end_token])), dim=0)
        return input_ids, label


def generate_text_until_end(
    input_text:str,
    model:GPT,
    tokenizer:ByteLevelBPETokenizer,
    max_length:int=100,
    device='cpu',
    temperature=1.0,
    top_k=3
):
    model = model.to(device)
    ids = tokenizer.encode(input_text).ids
    input_ids = torch.tensor(ids).unsqueeze(0).to(device)
    end_token_id = tokenizer.token_to_id("</s>")
    generated_ids = input_ids.flatten().clone()
    
    with torch.no_grad():
        while True:
            mask = generate_square_subsequent_mask(size=input_ids.size(1), device=device)
            output = model(input_ids=input_ids, attention_mask=mask)
            next_token_logits = output[:, -1, :] / temperature

            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
            top_k_probs = top_k_probs.squeeze()
            top_k_indices = top_k_indices.squeeze()
            
            next_token_id = torch.multinomial(top_k_probs, 1)
            next_token_id = top_k_indices[next_token_id]
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            input_ids = generated_ids.unsqueeze(0)
            
            if next_token_id == end_token_id or len(generated_ids) >= max_length:
                break
                
    generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    return generated_text
