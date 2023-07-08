import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import logging
import os

# Setting up logger
logging.basicConfig(
    filename = 'transformer_model.log',
    filemode = 'a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)
logger = logging.getLogger()

class PreprocessingTraining():
    def __init__(self, text, batch_size = 4, time_steps = 8):
        self.text = text,
        self.dataset_size = len(text)
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.batch = batch_size
        self.time = time_steps
        
        # tokenization and mapping
        self.str_to_int = {char:i for i, char in enumerate(self.vocab)}
        self.int_to_str = {i:char for i, char in enumerate(self.vocab)}
        
        # test, test, validation split
        self.train_text, self.val_text, self.test_text = self.train_test_validation_split()
        
    def encoding(self, st: str):
        return [self.str_to_int[char] for char in st]
    
    def decoding(self, li: list):
        return ''.join([self.int_to_str[i] for i in li])

    def train_test_validation_split(self):
        """
        Splits the dataset into train (81%), validation (9%), and test (10%).
    
        - First 90% is training+validation
        - Last 10% is test
        - From the 90%, 10% is taken as validation (i.e., 9% of total)
        """
    
        split_index_test = int(0.9 * len(self.text))
        train_val_text = self.text[:split_index_test]
        test_text = self.text[split_index_test:]

        split_index_val = int(0.9 * len(train_val_text))
        train_text = train_val_text[:split_index_val]
        val_text = train_val_text[split_index_val:]

        return train_text, val_text, test_text
    
    def get_batch_indices(self, split = 'train'):
        data = {'train': self.train_text, 'validation': self.val_text, 'test': self.test_text}[split]
        batch_indices = torch.randint(0, len(data)- self.time, (self.batch,))
        return batch_indices
    
    def get_batch(self, split = 'train'):
        batch_indices = self.get_batch_indices(split)
        data = {'train': self.train_text, 'validation': self.val_text, 'test': self.test_text}[split]
        x = torch.stack([torch.tensor(self.encoding(data[i:i+self.time])) for i in batch_indices])
        y = torch.stack([torch.tensor(self.encoding(data[i+1:i+self.time+1])) for i in batch_indices])
        return x, y
    
class TransformerModel(nn.Module):
    def __init__(self,vocab_size, channel_dim, context_window):
        super().__init__()
        # Each token in a batch is mapped to a learnable vector of size C
        self.token_embedding = nn.Embedding(vocab_size, channel_dim) # (B, T) -> (B, T, C)
        
        # Positional embedding tells the model where in the sequence the token is (position 0, 1, 2, ...) 
        self.position_embedding = nn.Embedding(context_window, channel_dim) 
        
        # Linear layers to project each token's embedding into Q (query), K (key), and V (value)
        self.query = nn.Linear(channel_dim, channel_dim)
        self.key = nn.Linear(channel_dim, channel_dim)
        self.value = nn.Linear(channel_dim, channel_dim)
        
        # Final linear projection from the output of attention to logits over vocabulary
        self.proj = nn.Linear(channel_dim, vocab_size)
        
        # save context window size (T) for masking later
        self.context_window = context_window
        
        # Lower triangular matrix to apply causal masking: token at position i can only attend to tokens at positions <=i
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window))) 
        # register_buffer in nn.Module basically tells pytorch that tril is part of model's state, but is not a trainable parameter, ie it is stored in model, but not updated during training
        
    def forward(self, x, targets = None):
        # x is of shape (B,T) 
        B,T = x.shape
        
        # embed tokens, so we basically look up that token's value and get C
        # B,T -> B,T,C
        token_emb = self.token_embedding(x)
        
        # create position embeddings
        # first we create position indices ranging from [0, T-1], and then we embed them
        pos_indices = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_indices) 
        
        # add token and position embeddings 
        x = token_emb + pos_emb
        
        # computing K, Q, V for all tokens in all sequences (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # computing attention scores for each token in a sequence Q@K(transpose) -> (B,T,T) 
        attn_scores = Q @ K.transpose(-2, -1) # Transpose last two dims of K to get dot product
        
        # scale scores to avoid very large values (stabilise gradients): divide by squared root of C
        attn_scores = attn_scores / (K.shape[-1] ** 0.5)
        
        # applying lower-triangular (causal) mask so model cannot cheat and know what token is occuring next
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # convert scores into probabilities across the sequence
        attn_weights = F.softmax(attn_scores, dim=-1) # (B,T,T)
        
        # use attn_weights to create weighted average of values
        context = attn_weights @ V  # (B, T, C)
        
        # Project the context vectors to logits over the vocabulary
        logits = self.proj(context) # (B, T, vocab_size)
        
        loss = None
        
        if targets is not None:
            # Reshape for cross entropy loss: (B, T, vocab_size) → (B*T, vocab_size)
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, input_ids, max_tokens_ahead = 100):
        """
        Autoregressively generate new tokens, starting from input_ids.

        input_ids: tensor of shape (B, T') where T' ≤ context_window
        Returns: tensor of shape (B, T' + max_tokens_ahead)
        """
        for _ in range(max_tokens_ahead):
            # Truncate to the last `context_window` tokens if too long
            input_condensed = input_ids[:, -self.context_window] # shape (B, T)
            
            # get logits from the model
            logits, _ = self.forward(input_condensed)
            
            # Take logits of last position only (last token generated)
            last_logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Convert logits to probability distribution
            probs = F.softmax(last_logits, dim=-1)  # (B, vocab_size)
            
            # Sample next token from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled token to input_ids
            input_ids = torch.cat((input_ids, next_token), dim=1)  # shape grows each step
        return input_ids
        
    
        
