# Character-Level Transformer: From Scratch

This project is a step-by-step implementation of a **decoder-only Transformer model** trained to predict the **next character** in a sequence, inspired by Andrej Karpathy's "Let's build GPT from scratch" tutorial.

Unlike typical NLP models trained on subwords or words, this model works at the **character level**, learning language structure purely from character sequences.

---

## Project Structure

| Stage                                | Description                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| **1. Bigram Model**                  | Simple baseline: Predict next character using frequency statistics           |
| **2. Token + Linear Model**          | Learn token embeddings and predict next character with a single linear layer |
| **3. Single Attention Head**         | Implement scaled dot-product attention (Q, K, V)                             |
| **4. Multi-Head Attention**          | Multiple heads for parallel attention + projection                           |
| **5. Transformer Block**             | Add feedforward layers, layer normalization, residuals                       |
| **6. Full Decoder-Only Transformer** | Stack multiple blocks and train end-to-end on character sequences            |
| **7. Evaluation**                    | Track training + validation loss, and generate sample text                   |

---

## Dataset

We use the **TinyShakespeare** dataset — a cleaned 1MB corpus of Shakespeare's plays.

- Source: [`tinyshakespeare.txt`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Size: ~1MB
- Task: Predict the next character in the sequence

---

## Why Character-Level Modeling?

Training transformer models on **character-level data** has several advantages, especially for developers working on **CPU-only systems**:

### Simpler Vocabulary

- Word-level or subword-level models often have vocabularies in the range of **10,000 to 50,000+ tokens**.
- In contrast, character-level modeling typically involves **50 to 150 unique characters** (including letters, punctuation, whitespace, etc.).
- This drastically **reduces memory usage** and **speeds up matrix operations** in the embedding and output layers.

### Lower Memory & Storage Needs

- Smaller vocab size means:
  - **Smaller embedding tables** (e.g., 128 × 65 instead of 128 × 50,000)
  - **Fewer output logits** to compute per token
- These savings are crucial for CPU-only systems without access to large VRAM or GPU acceleration.

### Lightweight Training

- **Smaller model architectures** (fewer heads, layers, and parameters)
- **Shorter sequences** (character-level sequences convey dense information per token)
- Training remains practical even on **older machines**, such as a 2015 MacBook Pro.

### Encourages Understanding

- With simpler inputs and smaller models, we can:
  - **Inspect every component** (embeddings, attention scores, etc.)
  - **Understand gradients, overfitting, and learning dynamics** without needing a cluster

> Character-level transformers provide an approachable, resource-friendly way to build real NLP systems while learning the inner workings of attention and language modeling from scratch.
