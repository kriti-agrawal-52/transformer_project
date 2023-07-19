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
        self.text = text
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
        
        # Positional embedding tells the model where in the sequence the token is (position 0, 1, 2, ...) by assigning a C-dim vector to each position in the sequence
        self.position_embedding = nn.Embedding(context_window, channel_dim)  # shape (T, C)
        
        # Linear layers to project each token's embedding into Q (query), K (key), and V (value)
        self.query = nn.Linear(channel_dim, channel_dim)
        self.key = nn.Linear(channel_dim, channel_dim)
        self.value = nn.Linear(channel_dim, channel_dim)
        # self.query = nn.Linear(C, C)
        # This line creates a learnable linear transformation layer that:
        # - Internally defines a weight matrix of shape (C_out, C_in) = (C, C)
        # - Internally defines a bias vector of shape (C,)
        # - Registers both weight and bias as trainable parameters (i.e., will be updated during backprop)
        # - The weight matrix is initialized randomly and will be used to transform each input vector

        
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
        # Add token embeddings and positional embeddings together
        # token_emb has shape: (B, T, C) → each token in each sequence has a C-dimensional embedding
        # pos_emb has shape: (T, C) → one unique C-dimensional vector per position in the sequence (0 to T-1)
        # Since all sequences in the batch share the same positions, the same pos_emb is added across the batch
        # PyTorch automatically broadcasts (T, C) to (1, T, C), and then adds it to (B, T, C)
        # Final result shape: (B, T, C) → now each token knows both its identity and its position

        
        # computing K, Q, V for all tokens in all sequences (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # What this line does under the hood:

        # Step 1: x is a 3D tensor of shape (B, T, C) — i.e., a batch of token sequences, each token is a C-dimensional vector

        # Step 2: For each token vector (size C), apply the linear transformation:
        #         result = input @ weightᵀ + bias
        #         where:
        #           - input is each token vector of shape (C,)
        #           - weight is shape (C, C), and is transposed internally by PyTorch
        #           - bias is added to each output vector

        # So the operation across all tokens in all sequences becomes:
        # Q = x @ Wᵀ + b
        # where:
        #   - x is (B, T, C)
        #   - Wᵀ is (C, C)
        #   - b is (C,) and is broadcasted over (B, T)
        #   - final Q shape = (B, T, C)
        # This lets each token learn what "query" it is sending out, based on its embedding.
        

        # computing attention scores for each token in a sequence Q@K(transpose) -> (B,T,T) 
        # Q (B, T, C) @ Kᵀ (B, C, T) → (B, T, T)
        attn_scores = Q @ K.transpose(-2, -1) # Transpose last two dims of K to get dot product
        
        # Scale the attention scores before applying softmax
        # Reason:
        # - Each row of Q and K is a C-dimensional vector (e.g., C = 64)
        # - The dot product between Q and K is computed by multiplying corresponding elements and summing:
        #     Q ⋅ K = q₁k₁ + q₂k₂ + ... + q_Ck_C
        # - So, as C increases, we're summing over more terms → the dot product value grows larger
        # - This leads to attention scores (Q @ Kᵀ) with larger magnitudes when C is high
        # - If attention scores are too large (e.g., [9.2, -3.5, 0.7, -4.2]), softmax becomes very sharp:
        #     softmax([9.2, -3.5, 0.7, -4.2]) → [0.999, 0.0001, 0.0008, 0.00005]
        #   → This is close to a one-hot vector — the model ends up attending to just one token and ignoring others
        #   → Each token aggregates information from only one other token → attention is no longer "distributed"
        # - This also makes gradients vanish during training, especially with stacked layers
        # - To fix this, we scale the scores by sqrt(C), which reduces their magnitude:
        #     Example: [1.1, -0.4, 0.08, -0.5] → softmax becomes [0.46, 0.12, 0.25, 0.17]
        # - This keeps the attention distribution diffused, allowing tokens to consider multiple others,
        #   and helps maintain stable gradients
        attn_scores = attn_scores / (K.shape[-1] ** 0.5)
        
        # applying lower-triangular (causal) mask so model cannot cheat and know what token is occuring next
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # For all positions in attn_scores where the mask value is 0 (i.e., above the diagonal),
        # replace them with -inf so that softmax will ignore them.
        
        # convert scores into probabilities across the sequence
        attn_weights = F.softmax(attn_scores, dim=-1) # shape of attn_scores is (B,T,T)
        
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
    
    def evaluate_validation_loss(self, get_batch_fn, eval_iters=20):
        """
        Computes average validation loss over `eval_iters` batches using forward passes only.
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
        Training loop with frequent validation checking, checkpointing, early stopping, and plotting.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        stale_checks = 0
        checkpoint_path = "transformer_checkpoint.pt"

        # Store losses
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
                val_loss = sum(self.evaluate_validation_loss(get_batch_fn)) / 20
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

        # 📊 Plot loss curves
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


        
def hyperparameter_search(raw_text, lrs=[1e-2], batch_sizes=[4], time_steps=[8]):
    results = []
    for bs in batch_sizes:
        for ts in time_steps:
            for lr in lrs:
                try:
                    print(f"Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")
                    logger.info(f"Starting Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")

                    prep = PreprocessingTraining(raw_text, batch_size=bs, time_steps=ts)
                    model = TransformerModel(prep.vocab_size, context_window=ts, channel_dim=32)

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

# == Driver Code ==
if __name__ == '__main__':
    try:
        with open('input.txt', 'r') as f:
            raw_text = f.read()
        logger.info("Successfully loaded input.txt")

        torch.manual_seed(1337)  # global seeding for reproducibility

        prep = PreprocessingTraining(raw_text)
        model = TransformerModel(prep.vocab_size, context_window=prep.time, channel_dim=32)

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

        
    
        
