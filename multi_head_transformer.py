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
        