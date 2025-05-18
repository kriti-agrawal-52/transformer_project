import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import logging
import os

# Setup logger
logging.basicConfig(
    filename="training.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

class PreprocessingTraining():
    def __init__(self, text, batch_size = 4, time_steps = 8):
        self.text = text
        self.dataset_size = len(self.text)
        self.vocab = sorted(set(self.text)) # number of distinct tokens in the dataset (for our model token is a character)
        self.vocab_size = len(self.vocab)
        self.Batch = batch_size # we create a batch of 4 chunks which can be processed paralllely by our language model
        self.Time = time_steps
        
        # Tokenization is the process of splitting text into units (tokens) like characters, words, or subwords,
        # and mapping each token to a unique numerical ID for model input.
        self.str_to_int = {char:index for index, char in enumerate(self.vocab)}
        self.int_to_str = {index:char for index, char in enumerate(self.vocab)}

        # Split the dataset once for reusability
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

    def get_batch_indices(self, split:'train'):
        torch.manual_seed(1337)
        data = {'train': self.train_text, 'validation': self.val_text, 'test': self.test_text}[split]
        # we are seeding our torch random generator so that when we reproduce or rerurn this code, we always get same random numbers
        batch_indices = torch.randint(0, len(data)-self.Time, (self.Batch,))
        # we are generating 4 random indices which can be any integers between 0 and len(data)-block_size which is Time
        return batch_indices

    def get_batch(self, split: 'train'):
        batch_indices = self.get_batch_indices(split)
        data = {
            'train': self.train_text,
            'validation': self.val_text,
            'test': self.test_text
        }[split]
        x = torch.stack([torch.tensor(self.encoding(data[i:i+self.Time])) for i in batch_indices])
        y = torch.stack([torch.tensor(self.encoding(data[i+1:i+self.Time+1])) for i in batch_indices])
        return x, y    
    
"""
BIGRAM MODEL – EXPLANATION AND INTUITION

This is a simple character-level bigram language model.
It learns to predict the next character based only on the current one — no memory of earlier tokens.

1. INPUT:
   - The model receives input as tokenized character sequences.
   - Shape: (B, T), where:
     B = batch size (number of sequences),
     T = time steps (tokens per sequence).
   - We are effectively using mini-batch stochastic gradient descent:
     At each training step, we randomly sample a new batch of (B, T)-shaped input and target sequences.
     This allows the model to update its parameters using a small subset of the training data at a time,
     making training faster and less memory-intensive than using the full dataset every time.
   - Using the full training set at each step (batch gradient descent) would be computationally expensive,
     slow to converge, and generally unnecessary for good generalization in deep learning.

2. MODEL:
   - A single embedding layer: nn.Embedding(vocab_size, vocab_size)
     This serves as a lookup table: for each token ID, we directly get logits over all possible next tokens.
   - The output of the embedding has shape (B, T, vocab_size).
     That means: for each token at each position in each sequence, we get one row of logits.

3. TARGET:
   - Also shape (B, T), these are the "ground truth" next tokens the model should predict.
   - Typically obtained by shifting the input by one position to the right.

4. LOSS (Cross-Entropy):
   - The model uses torch.nn.functional.cross_entropy to compare predicted logits to target tokens.
   - Logits are raw scores → cross_entropy automatically applies softmax inside.
   - Reshape needed:
       logits: (B, T, vocab_size) → (B*T, vocab_size)
       targets: (B, T) → (B*T)
   - Loss is the negative log-probability assigned to the correct token.

5. INTERPRETING INITIAL LOSS (Before Training):
   - At initialization, the logits are random and the softmax output is nearly uniform.
   - If vocab_size = 65, then each token has roughly 1/65 probability → log(1/65) ≈ -4.17
   - So initial loss should be around 4.17

   What loss tells us:
   - **Loss ≈ 4.17**: sanity check — model is untrained, predictions are near uniform.
   - **Loss << 4.17** before training: suspicious — possibly overfitting, data leak, or label issues.
   - **Loss >> 4.17**: model might be confidently wrong — spiky logits focused on incorrect tokens due to bad initialization.

6. TRAINING:
   - This model has no hidden layers, attention, or MLPs.
   - The only trainable component is the embedding matrix (vocab_size × vocab_size),
     which directly learns the conditional probability of every possible next character given the current one.
   - It essentially "memorizes" next-token frequencies.
   
   - Note: Since this is a bigram model, the context length T is not used for actual dependency modeling —
     the model only considers the current token to predict the next. So, increasing T does not improve model performance,
     it just helps parallelize more predictions per batch.
   - However, in more complex models like GPT or ChatGPT that use attention mechanisms,
     T defines how many past tokens the model can attend to — i.e., the **context window**.
     In such models, a larger T allows the model to capture long-range dependencies,
     and is often more important than batch size for performance.

7. WHY THIS MODEL IS IMPORTANT:
   - You can build a non-neural bigram model using raw token pair frequencies.
   - But this neural version introduces you to:
       - Training via gradient descent
       - Backpropagation
       - Autograd and loss functions
    which are foundational for scaling up to real transformer-based LLMs.

8. BIGRAM vs TRANSFORMER:
   - In bigram models, C (channel size) = vocab_size
   - In real transformers, C is a hidden dimension (e.g., 128 or 512), and only the final linear layer projects to vocab_size for logits.
   - Deeper models allow modeling longer-term dependencies, while this bigram model only looks one token back.
   
9. CHECKPOINTING & EARLY STOPPING:
   - Every `print_every` steps (e.g., every 100 steps), we compute the validation loss.
   - If this validation loss is smaller than the best one seen so far by at least `min_delta`,
     we consider it a significant improvement and **checkpoint** the model by saving its weights with `torch.save()`.
   - If no such improvement is seen for `patience` consecutive validation checks, we trigger **early stopping** —
     stopping training to avoid overfitting and wasted computation when the model is no longer learning.
"""

class BigramModel(nn.Module):
    # input is of shape (B,T)
    # B: batch size (number of sequences we're processing at once)
    # T: Time steps (length of each sequence)
    # output: shape (B, T, C)
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_lookup_table = nn.Embedding(vocab_size, vocab_size)

    # == forward pass ==
    def forward(self, x, targets = None):
        logits = self.token_lookup_table(x) # (B, T, C = vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # == autoregressive generation ==
    def generate(self,input_ids, max_tokens_ahead=100):
        for _ in range(max_tokens_ahead):
            logits, _ = self.forward(input_ids)
            # we need to get logits for only the last token, so that we can predict its next token till we generate max_tokens_ahead 
            logits = logits[:, -1, :] 
            # getting logit of only the last token in the input because we only need current token to predict next token, 
            # so we pluck logits of C length for last elements from sequence in each batch 
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, 1) # (B,1) in each of the batch dimensions, we will have a single prediction for what comes next
            # we concat this new token to out input 
            input_ids = torch.cat([input_ids, next_token], dim = 1) # so the new shape becomes (B, T+1)
        return input_ids

    def evaluate_validation_loss(self, get_batch_fn):
        self.eval() # Sets the model to evaluation mode: dropout is turned off and batchnorm layers use stored running statistics (instead of updating them)
        with torch.no_grad(): # telling the model that we are only inferencing, we dont need to calculate gradient of params
            xval, yval = get_batch_fn('validation')
            _, val_loss = self(xval, yval)
        return val_loss.item()

    
    # == Single training run with validation tracking ==
    def train_loop(
        self, 
        get_batch_fn, 
        prep, 
        steps = 1000, 
        print_every=100, 
        checkpoint_path = 'best_bigram.pt', 
        patience= 2,
        min_delta = 1e-3
    ):
        # patience: how many validation checks to wait
        # min_delta: required improvement to reset patience
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)
        
        self.train() # Sets the model to training mode: dropout is active and batchnorm layers update their running statistics

        train_losses, val_losses = [], []
        
        best_val = float('inf')
        max_stale_steps = patience * print_every
        stale = 0 # early stop counter
        
        for step in range(1, steps+1): # we could also start at 0, but 1 makes step logging more human readable
            try:
                xb, yb = get_batch_fn('train') # get batches from training data
                logits, loss = self(xb, yb) # forward pass
                optimizer.zero_grad() # clear old gradients 
                loss.backward() # backpropagation and calculation of gradients 
                optimizer.step() # updates weights (in this model that would be logit values in the token_lookup_table)
                train_losses.append(loss.item())
    
                # == Run validation and checkout every 'print_every' steps == 
                if step % print_every == 0 or step == steps:
                    val_loss = self.evaluate_validation_loss(get_batch_fn)
                    val_losses.append(val_loss)
                    print(f"[Step {step}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                    logger.info(f"Step {step}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
                    
                    # checkpoint model if improved
                    if best_val - val_loss > min_delta:
                         # Model improved → save checkpoint
                        best_val = val_loss
                        stale = 0
                        torch.save(self.state_dict(), checkpoint_path)
                        print(" ↑ New Best Model; Checkpoint saved.")
                    else:
                        # No improvement → increment early stopping counter
                        stale += print_every # stale counts how many times in a row val loss has not improved beyond min_delta
                        print(f"No improv. for {stale}/{max_stale_steps} steps/epochs")
    
                    # checking if we need to do early stop
                    if stale >= max_stale_steps: # Early stopping triggered if no improvement for `patience` validation checks
                        print(">>> Early stopping triggered.")
                        logger.warning(">>> Early stopping triggered.")
                        break
                    
                    # trying a generation to see how model is performing at this step during training
                    prompt = "Th"
                    input_ids = torch.tensor([prep.encoding(prompt)])
                    sample = self.generate(input_ids, 40)
                    decoded = prep.decoding(sample[0].tolist())
                    print(f"Sample generation for Th at step {step}: {decoded}\n")
    
                    self.train() # back to train mode
            except Exception as e:
                logger.error(f"Error during training at step : {step}: {e}")

        # == plot training vs validation loss ==
        try:
            plt.figure(figsize=(10, 6))  # Make the plot larger for clarity

            # Plot training loss (dense line)
            plt.plot(train_losses, label='Training Loss', color='blue', linewidth=1.5)

            # Plot validation loss (every `print_every` steps)
            val_steps = list(range(print_every - 1, len(train_losses), print_every))
            plt.plot(val_steps, val_losses, 'o-', label='Validation Loss', color='orange', linewidth=2, markersize=5)

            # Axis labels and title
            plt.xlabel("Training Step", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Training vs. Validation Loss Curve", fontsize=14)

            # Grid and legend
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right', fontsize=11)

            # Save to file
            plt.tight_layout()
            plt.savefig('loss_curves.png')
            plt.show()  # Optional: only works if in a GUI or notebook
        except Exception as e:
            logger.error(f"Error plotting training curves {e}")

# == hyperparameter tuning function ==
def hyperparameter_search(raw_text, lrs = [1e-2], batch_sizes = [4], time_steps = [8]):
    results = []
    for bs in batch_sizes:
        for ts in time_steps:
            for lr in lrs:
                try:
                    print(f"Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")
                    logger.info(f"Starting Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")
                    prep = PreprocessingTraining(raw_text, batch_size=bs, time_steps=ts)
                    model = BigramModel(prep.vocab_size)
                    model.train_loop(prep.get_batch, prep, steps=1000, print_every=100)
                    val_loss = model.evaluate_validation_loss(prep.get_batch)
                    results.append({
                        'batch_size': bs,
                        'time_steps': ts,
                        'learning_rate': lr,
                        'val_loss': val_loss # final validation loss for the model
                    })
                    logger.info(f"Completed run: bs={bs}, ts={ts}, lr={lr}, val_loss={val_loss:.4f}")
                except Exception as e:
                    logger.error(f"Error in tuning run (bs={bs}, ts={ts}, lr={lr}): {e}")
    return results


if __name__ == '__main__':
    try: 
        with open('input.txt', 'r') as f:
            raw_text = f.read()
        logger.info("Successfully loaded input.txt")
    
        # Initialize preprocessing and model
        prep = PreprocessingTraining(raw_text)
        model = BigramModel(prep.vocab_size)
    
        # forward pass before training (sanity check)
        xb, yb = prep.get_batch("train")
        _, loss_before = model(xb, yb)
        print(f"Initial loss should be ideally ~4.17 before training. Loss before training: {loss_before.item():.4f}")
        logger.info(f"Initial loss before training: {loss_before:.4f}")
    
        # Flag to control training
        should_train = True
        if should_train:
            model.train_loop(prep.get_batch, prep, steps=1000, print_every=100)
    
        # Evaluate model after training
        _, loss_after = model(xb, yb)
        print(f"Post-training loss: {loss_after:.4f}")
        logger.info(f"Post-training loss: {loss_after:.4f}")
    
        # Evaluate model on test data
        x_test, y_test = prep.get_batch("test")
        _, test_loss = model(x_test, y_test)
        print(f"Test set loss: {test_loss:.4f}")
        logger.info(f"Test set loss: {test_loss:.4f}")
    
        # Text generation from a prompt
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
        should_tune = True # set True to run tuning
        if should_tune:
            lrs = [1e-2, 5e-3]
            batch_sizes = [4, 8]
            time_steps = [8, 16]
            tuning_results = hyperparameter_search(raw_text, lrs, batch_sizes, time_steps)
            
            # Display top configs
            sorted_results = sorted(tuning_results, key=lambda x: x['val_loss'])
            print("\nTop 3 Hyperparameter Configurations:")
            for config in sorted_results[:3]:
                print(config)
    except Exception as e:
        logger.critical(f"Unhandled error during hyperparameter tuning: {e}")