import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import logging
import os
from transformers import AutoTokenizer


# Setting up logger
logging.basicConfig(
    filename='transformer_model_wordseq.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


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
        self.query = nn.Linear(embed_dim, embed_dim, bias = False)
        self.key = nn.Linear(embed_dim, embed_dim, bias = False)
        self.value = nn.Linear(embed_dim, embed_dim, bias = False)

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
    def __init__(self, text, batch_size=4, time_steps=64):
        self.text = text
        self.tokenizer = tokenizer
        self.batch = batch_size
        self.time = time_steps
        
        logger.info("Tokenizing the entire dataset...")
        
        self.all_token_ids = self.tokenizer.encode(self.text)
        logger.info(f"Tokenization complete. Total tokens: {len(self.all_token_ids)}")
        
        # KEY CHANGE: Get vocab size from the tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # KEY CHANGE: Split tokenized data
        self._split_tokenized_data() # New method to split token IDs

        self.train_text, self.val_text, self.test_text = self.train_test_validation_split()
    
    def _split_tokenized_data(self, train_val_ratio=0.9, val_ratio_of_train_val=0.1):
        """
        Splits `self.all_token_ids` into train, validation, and test sets (as tensors).
        The logic mirrors the original script's 81% train, 9% val, 10% test split.
        """
        n_tokens = len(self.all_token_ids)
        
        # Determine split point for test data (10% for test)
        train_val_idx_end = int(n_tokens * train_val_ratio) # First 90% for train+val
        
        all_tokens_tensor = torch.tensor(self.all_token_ids, dtype=torch.long)

        train_val_data_tokens = all_tokens_tensor[:train_val_idx_end]
        self.test_tokens = all_tokens_tensor[train_val_idx_end:]

        # Split train_val_data further into train and validation
        # (90% of train_val for train, 10% of train_val for val)
        val_idx_start = int(len(train_val_data_tokens) * (1 - val_ratio_of_train_val))
        
        self.train_tokens = train_val_data_tokens[:val_idx_start]
        self.val_tokens = train_val_data_tokens[val_idx_start:]

        logger.info(f"Train tokens: {len(self.train_tokens)}, Val tokens: {len(self.val_tokens)}, Test tokens: {len(self.test_tokens)}")
        if len(self.train_tokens) == 0 or len(self.val_tokens) == 0 or len(self.test_tokens) == 0:
            logger.warning("One or more data splits are empty. Check dataset size and split ratios.")
            
    def encode_string(self, s: str): # Renamed from 'encoding' for clarity
        """Encodes a string to a list of token IDs."""
        return self.tokenizer.encode(s)
    
    def decode_ids(self, ids: list): # Renamed from 'decoding' for clarity
        """Decodes a list of token IDs back to a string."""
        return self.tokenizer.decode(ids)

    def get_batch(self, split='train'):
        """
        Generates a batch of input sequences (x) and target sequences (y)
        from the specified tokenized data split (train, validation, or test).
        """
        if split == 'train':
            data_tokens = self.train_tokens
        elif split == 'validation': # Changed from 'validation' to 'val' to match attribute name
            data_tokens = self.val_tokens
        elif split == 'test':
            data_tokens = self.test_tokens
        else:
            raise ValueError(f"Unknown split type: {split}")

        # Ensure there's enough data to form a complete sequence of `time_steps + 1`.
        max_start_idx = len(data_tokens) - self.time_steps - 1
        if max_start_idx < 0:
             raise ValueError(
                 f"Dataset split '{split}' is too small ({len(data_tokens)} tokens) "
                 f"for time_steps={self.time_steps}. Need at least {self.time_steps + 1} tokens. "
                 f"Consider a smaller time_steps, larger dataset, or different split ratios."
            )
        
        # Randomly select `batch_size` starting indices.
        # Ensure randint upper bound is exclusive, so max_start_idx + 1
        ix = torch.randint(0, max_start_idx + 1, (self.batch_size,))

        # Create input sequences (x) and target sequences (y) from token IDs.
        x = torch.stack([data_tokens[i : i + self.time_steps] for i in ix])
        y = torch.stack([data_tokens[i + 1 : i + self.time_steps + 1] for i in ix])
        
        # Note: Moving to device will be handled in the training loop / evaluation function
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
    def __init__(self, vocab_size, channel_dim, context_window, num_heads=8, num_layers=6):
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

    def forward(self, x_indices, targets=None):
        # x: (B, T) where T = context length (sequence of token indices)

        B, T = x_indices.shape
        
        if T > self.context_window:
             logger.warning(f"Input sequence length {T} exceeds model context window {self.context_window}. Truncating.")
             x_indices = x_indices[:, :self.context_window] # Simple truncation
             T = self.context_window
             if targets is not None:
                 targets = targets[:, :self.context_window]

        # Step 1: Embed tokens → (B, T, C)
        token_emb = self.token_embedding(x_indices)

        # Step 2: Embed positions (broadcasts across batch)
        pos_indices = torch.arange(T, device=x_indices.device)
        pos_emb = self.position_embedding(pos_indices)  # (T, C)

        # Step 3: Combine token + position embeddings
        x = token_emb + pos_emb  # (B, T, C)
        
        for block in self.transformer_blocks:
            x = block(x)  

        x = self.ln_f(x)                                         # Final layer norm
        logits = self.proj(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Ensure targets are also on the same device as logits
            targets = targets.to(logits.device)
            B_logits, T_logits, C_logits = logits.shape # C_logits is vocab_size
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
            
        return logits, loss
    
    @torch.no_grad()
    def evaluate_validation_loss(self, prep_obj, eval_iters=20):
        """
        Computes average validation loss over `eval_iters` mini-batches.
        """
        self.eval()
        losses = []
        for _ in range(eval_iters):
            xb, yb = prep_obj.get_batch('validation')
            _, loss = self(xb, yb)
            losses.append(loss.item())
        self.train()
        
        return sum(losses) / len(losses) if losses else float('nan')
    
    def train_loop(self, prep_obj, steps=2000, val_check_every=50, patience=4, min_delta=1e-4, lr=5e-3):
        """
        Training loop with early stopping, validation checks, and plotting.
        """
        self.to(DEVICE) # Move model to device
        logger.info(f"Starting training on {DEVICE}")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        stale_checks = 0
        checkpoint_path = "transformer_checkpoint_wordseq.pt"

        train_losses = []
        val_loss_dict = {}
        self.train()

        for step in range(1, steps + 1):
            xb, yb = prep_obj.get_batch('train')
            xb, yb = xb.to(DEVICE), yb.to(DEVICE) # Move batch to device
            logits, loss = self(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if step % val_check_every == 0 or step == steps: # Also check at the very last step
                val_loss = self.evaluate_validation_loss(prep_obj, eval_iters=20) # Pass prep_obj
                val_loss_dict[step] = val_loss

                logger.info(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                print(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")


                if val_loss < best_val_loss - min_delta : # Check if val_loss is better
                    best_val_loss = val_loss
                    stale_checks = 0
                    torch.save(self.state_dict(), checkpoint_path)
                    logger.info(f" ↑ New Best Model (Val Loss: {best_val_loss:.4f}); Checkpoint saved to {checkpoint_path}")
                    print(" ↑ New Best Model; Checkpoint saved.")
                else:
                    stale_checks += 1
                    logger.info(f" No improvement for {stale_checks}/{patience} validation checks.")
                    print(f" No improvement for {stale_checks}/{patience} validation checks.")

                if stale_checks >= patience:
                    logger.warning(">>> Early stopping triggered.")
                    print(">>> Early stopping triggered.")
                    break
        
        if train_losses and val_loss_dict: # Plotting
            plt.figure(figsize=(12, 7))
            plt.plot(train_losses, label='Training Loss (per step)', color='lightblue', alpha=0.7, linewidth=1)
            
            if len(train_losses) >= val_check_every: # Smoothed training loss
                smoothed_train_loss = [sum(train_losses[i:i+val_check_every])/val_check_every for i in range(0, len(train_losses) - val_check_every + 1, val_check_every)]
                steps_for_smoothed = list(range(val_check_every, len(train_losses)+1, val_check_every))[:len(smoothed_train_loss)]
                plt.plot(steps_for_smoothed, smoothed_train_loss, label=f'Smoothed Training Loss (avg over {val_check_every} steps)', color='blue', linewidth=2)

            plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), 'o-', label='Validation Loss', color='orange', linewidth=2, markersize=5)
            
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('loss_curve_modified.png')
            logger.info("Loss curve saved to loss_curve_modified.png")
            # plt.show() # plt.show() can block execution in some environments, make it optional
        else:
            logger.info("No data for plotting loss curves.")


        return train_losses, val_loss_dict
    
    @torch.no_grad() # Added decorator
    def generate(self, input_ids, max_tokens_ahead = 100, temperature=1.0, top_k=None): # Added temp and top_k
        """
        Autoregressively generate new tokens, starting from input_ids.
        input_ids: tensor of shape (B, T') where T' ≤ context_window
        Returns: tensor of shape (B, T' + max_tokens_ahead)
        """
        self.eval() # Set to eval mode
        # device = next(self.parameters()).device # Get model's device
        input_ids = input_ids.to(DEVICE) # Ensure input_ids are on the correct device

        for _ in range(max_tokens_ahead):
            # Truncate to the last `context_window` tokens if too long
            if input_ids.size(1) > self.context_window:
                input_condensed = input_ids[:, -self.context_window:]
            else:
                input_condensed = input_ids
            
            logits, _ = self.forward(input_condensed) # self() calls forward
            
            last_logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature > 0: # temperature=0 can lead to issues with multinomial if not handled
                 last_logits = last_logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -float('Inf') # Apply top-k filtering
            
            probs = F.softmax(last_logits, dim=-1)
            
            # Sample next token
            # Avoid issues with temperature=0 if it results in all zeros after softmax (unlikely with top_k)
            if temperature == 0: # Deterministic: take the most probable token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
    
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
                    model = TransformerModel(prep.vocab_size, context_window=ts, channel_dim=64, num_heads=8)

                    model.train_loop(prep.get_batch, prep, steps=1000, val_check_every=20, patience=6, lr=lr)

                    val_loss = model.evaluate_validation_loss(prep.get_batch)
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
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        prep = PreprocessingTraining(raw_text, tokenizer = tokenizer)
        model = TransformerModel(prep.vocab_size, context_window=prep.time, channel_dim=128, num_heads=8, num_layers=6)

        xb, yb = prep.get_batch("train")
        _, loss_before = model(xb, yb)
        print(f"Initial loss should be ideally ~4.17 before training. Loss before training: {loss_before.item():.4f}")
        logger.info(f"Initial loss before training: {loss_before:.4f}")

        should_train = True
        if should_train:
            model.train_loop(prep.get_batch, prep, steps=2000, val_check_every=50, patience=4)

        _, loss_after = model(xb, yb)
        print(f"Post-training loss: {loss_after:.4f}")
        logger.info(f"Post-training loss: {loss_after:.4f}")

        x_test, y_test = prep.get_batch("test")
        _, test_loss = model(x_test, y_test)
        print(f"Test set loss: {test_loss:.4f}")
        logger.info(f"Test set loss: {test_loss:.4f}")

        prompt = 'Romeo'
        input_ids = torch.tensor([prep.encoding(prompt)])
        generated_ids = model.generate(input_ids, 20)
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
        should_tune = False
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

        