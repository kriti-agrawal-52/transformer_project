import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import logging
import os

# Setting up logger
logging.basicConfig(
    filename='transformer_model.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

"""
======================== MULTI-HEAD ATTENTION: DETAILED EXPLANATION ========================

OBJECTIVE:
Multi-head attention enables the model to attend to different parts of the input sequence
simultaneously, from multiple perspectives. It is a foundational component of the transformer
architecture, designed to capture complex dependencies between tokens — like word order,
semantic relations, long-range context — by splitting the attention mechanism into multiple
parallel attention "heads".

Each attention head receives a distinct view of the input embeddings and computes its own
attention distribution. The outputs from all heads are then concatenated and combined into
a single representation that captures richer contextual information.

--------------------------------------------------------------------------------------------
TERMINOLOGY AND SHAPES:

Let:
- B = batch size
- T = sequence length (context window)
- C = embedding dimension (also called channel dimension)
- H = number of attention heads
- D = dimension per head, such that D = C // H

Example: B = 4, T = 16, C = 64, H = 4 → then D = 16

--------------------------------------------------------------------------------------------
STEP-BY-STEP EXPLANATION:

1. Input Embeddings:
   The input `x` to the attention layer is a tensor of shape (B, T, C).
   Each token in the sequence is already embedded into a C-dimensional vector.

2. Linear Projections for Query, Key, Value:
   We apply three separate learnable linear layers to the input `x`:
       - Q = x @ W_q
       - K = x @ W_k
       - V = x @ W_v
   These layers map the original embedding space into three new spaces
   (queries, keys, and values), each still of shape (B, T, C).

   Note: We use a single shared linear layer for each of Q, K, V and
   then split their outputs into heads, instead of using separate Q/K/V
   for each head. This makes the implementation efficient.

3. Reshape and Transpose for Multi-Headed Computation:
   We split the C-dimensional Q/K/V vectors into H heads:
       - Q: (B, T, C) → (B, H, T, D)
       - K: (B, T, C) → (B, H, T, D)
       - V: (B, T, C) → (B, H, T, D)

   This is done using:
       Q.view(B, T, H, D).transpose(1, 2)
   So now, each attention head will process a sequence of T tokens,
   where each token is represented by a D-dimensional vector (not C anymore).

   Example:
   - If C = 64 and H = 4 → D = 16
   - Each token’s 64-dim vector is split into 4 chunks of 16 dims each
   - Each head focuses on a different 16-dim subspace

4. Scaled Dot-Product Attention (per head):
   For each head independently, we compute:
       - Attention scores = Q @ Kᵀ → shape: (B, H, T, T)
       - Scale the scores by √D to prevent softmax from becoming too sharp
       - Apply causal mask if using decoder (to prevent attending to future tokens)
       - Apply softmax to get attention weights
       - Multiply attention weights with V:
             context = softmax(QKᵀ / √D) @ V → shape: (B, H, T, D)

   This gives us the attention output for each head.

5. Concatenate Heads:
   We transpose and reshape the output from all heads back to original form:
       - context: (B, H, T, D) → (B, T, H*D) = (B, T, C)

   This merges the outputs from all attention heads into a single tensor.

6. Final Output Projection:
   We apply one more linear layer (self.out_proj) to map the concatenated
   attention output back to the original embedding dimension C:
       - output = context @ W_o → shape: (B, T, C)

   This output is then either passed to the next layer or used for further processing
   like feedforward layers or final classification logits.

--------------------------------------------------------------------------------------------
WHY NOT USE SEPARATE Q/K/V LINEAR LAYERS FOR EACH HEAD?

While it's conceptually possible to define separate Linear(C, D) layers for each head
(i.e., one Q/K/V per head), this is:
   - Less efficient: cannot leverage batched matrix multiplication
   - Requires more parameters: H times more weights
   - Harder to implement and debug

Instead, we use one shared Linear(C, C) layer for Q, K, and V each, and **then split the
output across heads** via reshaping. This gives us the same effect as having different heads,
but with greater computational efficiency and fewer parameters.

--------------------------------------------------------------------------------------------
REQUIREMENT:
The total embedding dimension C must be divisible by the number of heads H.
Otherwise, the reshape into (B, H, T, D) will fail. This is enforced using:

    assert C % H == 0

--------------------------------------------------------------------------------------------
IN SUMMARY:
- Input: x ∈ (B, T, C)
- Shared Q/K/V: Linear(C, C) → (B, T, C)
- Split: (B, T, C) → (B, H, T, D)
- Attention per head: softmax(QKᵀ / √D) @ V → (B, H, T, D)
- Merge heads: (B, H, T, D) → (B, T, C)
- Output projection: Linear(C, C)

The result is a rich, context-aware representation for each token, 
where multiple perspectives (heads) have contributed to its final form.

============================================================================================
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, context_window):
        super().__init__()
        
        # Each token will be represented by a vector of size `embed_dim`
        # We'll split it into `num_heads` smaller heads to capture attention from different perspectives
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        # The assert statement is used to enforce that a condition must be true at runtime.
        # If the condition fails, Python raises an AssertionError 
        
        self.num_heads = num_heads                 # Total number of heads
        self.head_dim = embed_dim // num_heads     # Size of each individual head (D)

        # Learnable linear projections for queries, keys, and values
        # They all map from (embed_dim → embed_dim), internally split later into heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # After attention, we combine the heads and project back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Lower triangular matrix for causal masking (ensures token i only attends to tokens ≤ i)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        # x: shape (B, T, C) where:
        # B = batch size, T = time/context length, C = embedding dimension

        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # === Step 1: Linear projections ===
        # Output shape: (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # === Step 2: Reshape into multi-heads ===
        # From (B, T, C) → (B, H, T, D)
        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # === Step 3: Compute attention scores ===
        # Dot product: Q @ K^T → (B, H, T, T)
        attn_scores = Q @ K.transpose(-2, -1)

        # Scale scores by sqrt(D) for gradient stability
        attn_scores = attn_scores / (D ** 0.5)

        # Apply causal mask (prevent attending to future tokens)
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax across last dimension to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # === Step 4: Compute weighted sum of values ===
        # Weighted sum: (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        attn_output = attn_weights @ V

        # === Step 5: Combine heads ===
        # Reshape: (B, H, T, D) → (B, T, H*D = C)
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.out_proj(out)
    
# === Preprocessing Class ===
class PreprocessingTraining():
    def __init__(self, text, batch_size=4, time_steps=8):
        self.text = text
        self.dataset_size = len(text)
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.batch = batch_size
        self.time = time_steps

        self.str_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_str = {i: char for i, char in enumerate(self.vocab)}

        self.train_text, self.val_text, self.test_text = self.train_test_validation_split()

    def encoding(self, st: str):
        return [self.str_to_int[char] for char in st]

    def decoding(self, li: list):
        return ''.join([self.int_to_str[i] for i in li])

    def train_test_validation_split(self):
        split_index_test = int(0.9 * len(self.text))
        train_val_text = self.text[:split_index_test]
        test_text = self.text[split_index_test:]

        split_index_val = int(0.9 * len(train_val_text))
        train_text = train_val_text[:split_index_val]
        val_text = train_val_text[split_index_val:]

        return train_text, val_text, test_text

    def get_batch_indices(self, split='train'):
        data = {'train': self.train_text, 'validation': self.val_text, 'test': self.test_text}[split]
        return torch.randint(0, len(data) - self.time, (self.batch,))

    def get_batch(self, split='train'):
        batch_indices = self.get_batch_indices(split)
        data = {'train': self.train_text, 'validation': self.val_text, 'test': self.test_text}[split]
        x = torch.stack([torch.tensor(self.encoding(data[i:i + self.time])) for i in batch_indices])
        y = torch.stack([torch.tensor(self.encoding(data[i + 1:i + self.time + 1])) for i in batch_indices])
        return x, y
    
class TransformerBlock(nn.Module):
    def __init__(self, channel_dim, num_heads, context_window):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.attn = MultiHeadAttention(channel_dim, num_heads, context_window)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.ffn = nn.Sequential(
            nn.Linear(channel_dim, 4 * channel_dim),
            nn.ReLU(),  # or GELU for GPT-style
            nn.Linear(4 * channel_dim, channel_dim)
        )

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # Feedforward with residual
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, channel_dim, context_window, num_heads=4, num_layers=4):
        super().__init__()

        # Embedding table: maps token indices to vectors of size channel_dim
        self.token_embedding = nn.Embedding(vocab_size, channel_dim)

        # Positional embedding: one vector per position up to context_window
        self.position_embedding = nn.Embedding(context_window, channel_dim)
        
        # Stack of Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channel_dim, num_heads, context_window)
            for _ in range(num_layers)
        ])

        # Final normalization and output projection
        self.ln_f = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, vocab_size)

        self.context_window = context_window

    def forward(self, x, targets=None):
        # x: (B, T) where T = context length (sequence of token indices)

        B, T = x.shape

        # Step 1: Embed tokens → (B, T, C)
        token_emb = self.token_embedding(x)

        # Step 2: Embed positions (broadcasts across batch)
        pos_indices = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_indices)  # (T, C)

        # Step 3: Combine token + position embeddings
        x = token_emb + pos_emb  # (B, T, C)
        
        for block in self.transformer_blocks:
            x = block(x)  

        x = self.ln_f(x)                                         # Final layer norm
        logits = self.proj(x)                                    # (B, T, vocab_size)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    def evaluate_validation_loss(self, get_batch_fn, eval_iters=20):
        """
        Computes average validation loss over `eval_iters` mini-batches.
        """
        self.eval()
        losses = []
        with torch.no_grad():
            for _ in range(eval_iters):
                xb, yb = get_batch_fn('validation')
                _, loss = self(xb, yb)
                losses.append(loss.item())
        self.train()
        return losses
    
    def train_loop(self, get_batch_fn, prep, steps=1000, val_check_every=20, patience=6, min_delta=1e-4, lr=1e-3):
        """
        Training loop with early stopping, validation checks, and plotting.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        stale_checks = 0
        checkpoint_path = "transformer_checkpoint.pt"

        train_losses = []
        val_loss_dict = {}

        for step in range(1, steps + 1):
            xb, yb = get_batch_fn('train')
            logits, loss = self(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if step % val_check_every == 0:
                val_loss = sum(self.evaluate_validation_loss(get_batch_fn)) / eval_iters
                val_loss_dict[step] = val_loss

                print(f"[Step {step}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    stale_checks = 0
                    torch.save(self.state_dict(), checkpoint_path)
                    print(" ↑ New Best Model; Checkpoint saved.")
                else:
                    stale_checks += 1
                    print(f" No improvement for {stale_checks}/{patience} validation checks.")

                if stale_checks >= patience:
                    print(">>> Early stopping triggered.")
                    break

        # Plot training vs validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=1)
        plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), 'o-', label='Validation Loss', color='orange', linewidth=2, markersize=4)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('loss_curve.png')
        plt.show()

        return train_losses, val_loss_dict
    
    def generate(self, input_ids, max_tokens_ahead = 100):
        """
        Autoregressively generate new tokens, starting from input_ids.

        input_ids: tensor of shape (B, T') where T' ≤ context_window
        Returns: tensor of shape (B, T' + max_tokens_ahead)
        """
        for _ in range(max_tokens_ahead):
            # Truncate to the last `context_window` tokens if too long
            input_condensed = input_ids[:, -self.context_window:] # shape (B, T)
            
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

# === Hyperparameter Search ===
def hyperparameter_search(raw_text, lrs=[1e-2], batch_sizes=[4], time_steps=[8]):
    results = []
    for bs in batch_sizes:
        for ts in time_steps:
            for lr in lrs:
                try:
                    print(f"Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")
                    logger.info(f"Starting Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")

                    prep = PreprocessingTraining(raw_text, batch_size=bs, time_steps=ts)
                    model = TransformerModel(prep.vocab_size, context_window=ts, channel_dim=32, num_heads=4)

                    model.train_loop(prep.get_batch, prep, steps=1000, val_check_every=20, patience=6, lr=lr)

                    val_loss = sum(model.evaluate_validation_loss(prep.get_batch)) / 20
                    results.append({
                        'batch_size': bs,
                        'time_steps': ts,
                        'learning_rate': lr,
                        'val_loss': val_loss
                    })
                    logger.info(f"Completed run: bs={bs}, ts={ts}, lr={lr}, val_loss={val_loss:.4f}")
                except Exception as e:
                    logger.error(f"Error in tuning run (bs={bs}, ts={ts}, lr={lr}): {e}")
    return results

# === Main Execution ===
if __name__ == '__main__':
    try:
        with open('input.txt', 'r') as f:
            raw_text = f.read()
        logger.info("Successfully loaded input.txt")

        torch.manual_seed(1337)

        prep = PreprocessingTraining(raw_text)
        model = TransformerModel(prep.vocab_size, context_window=prep.time, channel_dim=32, num_heads=4, num_layers=4)

        xb, yb = prep.get_batch("train")
        _, loss_before = model(xb, yb)
        print(f"Initial loss should be ideally ~4.17 before training. Loss before training: {loss_before.item():.4f}")
        logger.info(f"Initial loss before training: {loss_before:.4f}")

        should_train = True
        if should_train:
            model.train_loop(prep.get_batch, prep, steps=1000, val_check_every=20, patience=6)

        _, loss_after = model(xb, yb)
        print(f"Post-training loss: {loss_after:.4f}")
        logger.info(f"Post-training loss: {loss_after:.4f}")

        x_test, y_test = prep.get_batch("test")
        _, test_loss = model(x_test, y_test)
        print(f"Test set loss: {test_loss:.4f}")
        logger.info(f"Test set loss: {test_loss:.4f}")

        prompt = 'Th'
        input_ids = torch.tensor([prep.encoding(prompt)])
        generated_ids = model.generate(input_ids, 100)
        print("\nGenerated Text:\n" + prep.decoding(generated_ids[0].tolist()))

    except FileNotFoundError:
        logger.critical("input.txt not found. Make sure the dataset is downloaded.")
    except ValueError as e:
        logger.critical(f"Value error during training: {e}")
    except KeyError as e:
        logger.critical(f"Key error encountered (likely in vocab mapping): {e}")
    except RuntimeError as e:
        logger.critical(f"Runtime error (possibly tensor mismatch or CUDA error): {e}")
    except Exception as e:
        logger.critical(f"Unhandled error in driver training code: {e}")
        
    # == Optional: Hyperparameter Search Execution ==
    try:
        should_tune = True
        if should_tune:
            lrs = [1e-2, 5e-3]
            batch_sizes = [4, 8]
            time_steps = [8, 16]
            tuning_results = hyperparameter_search(raw_text, lrs, batch_sizes, time_steps)

            sorted_results = sorted(tuning_results, key=lambda x: x['val_loss'])
            print("\nTop 3 Hyperparameter Configurations:")
            for config in sorted_results[:3]:
                print(config)
    except Exception as e:
        logger.critical(f"Unhandled error during hyperparameter tuning: {e}")

        