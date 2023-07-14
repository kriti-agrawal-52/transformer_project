# **Introduction**

This repository contains a step-by-step implementation of a decoder-only Transformer model trained to predict the next token in a sequence. It is inspired by Andrej Karpathy's "Let's build GPT from scratch" tutorial, but significantly extended with additional features and improvements.

The code progresses from a basic bigram language model to a full multi-head, multi-layer Transformer architecture.

### **Key Enhancements**

- **Bigram and Transformer Models**: Includes implementations of a bigram model, a single-head single-layer Transformer, and a multi-head multi-layer Transformer.
- **Token Granularity**: Supports both character-level and word-level tokenization, allowing experimentation with different input granularities.
- **Training Features**: Adds evaluation metrics, checkpointing, and early stopping to make training more robust.
- **Text Generation Improvements**: Implements generation controls like temperature and top-k sampling to allow more flexible output generation.
- **Hyperparameter Tuning**: Enables configurable search over batch size, learning rate, context window, and other parameters to optimize training.

## **Bigram Language Model**

### **Model Architecture**

The bigram model learns to predict the next character based solely on the current one.

- **Core Logic**: It uses an embedding layer (`nn.Embedding`) where each character's embedding directly represents the logits (prediction scores) for the next possible characters.

### **Training Method**

The model is trained using **mini-batch stochastic gradient descent**.

- **Batching**: Random mini-batches of data are used for each training step.
- **Optimizer**: The `AdamW` optimizer updates the model's parameters.
- **Loss Function**: `Cross-entropy loss` is used to measure the difference between predicted and actual next characters.
- **Regularization**: Training incorporates **checkpointing** (saving the best model) and **early stopping** (halting training if validation loss plateaus) to prevent overfitting.

### **Code Structure**

- **`PreprocessingTraining` Class**:

  - Handles dataset loading, character-to-integer mapping (tokenization), and splitting data into training, validation, and test sets.
  - Provides methods to generate random mini-batches for training.

- **`BigramModel` Class**:

  - Defines the neural network with its single embedding layer.
  - Includes methods for the forward pass, text generation, and validation loss evaluation.
  - The `train_loop` method orchestrates the training process, manages optimization, and implements early stopping and checkpointing.

- **`hyperparameter_search` Function**:
  - An optional utility to explore different combinations of learning rates, batch sizes, and time steps to identify optimal training configurations.

## **Single-Head, Single-Layer Transformer Language Model**

This repository presents a foundational character-level Transformer model with a single attention head and one layer. It's built to illustrate the core mechanisms of the Transformer architecture for next-character prediction.

### **Model Architecture**

This Transformer model predicts the next character by understanding the context of preceding characters.

- **Embedding Layers**: It utilizes both `token embeddings` (mapping each character to a learnable vector) and `positional embeddings` (indicating each character's position within a sequence). These are summed to give the model awareness of both character identity and order.
- **Self-Attention Mechanism**:
  - `Query (Q)`, `Key (K)`, and `Value (V)` linear layers transform the combined token and positional embeddings.
  - Attention scores are computed by multiplying `Q` and `K` (scaled by the square root of the channel dimension).
  - A `causal mask` (lower-triangular matrix) is applied to ensure that each character can only attend to previous characters, preventing it from "seeing" future information.
  - `Softmax` converts these scores into attention weights, which are then used to create a weighted sum of `V` to form the `context vector`.
- **Projection Layer**: A final linear layer projects the context vectors to `logits` over the entire vocabulary, indicating the model's prediction for the next character.

### **Training Method**

The model is trained using **mini-batch stochastic gradient descent**.

- **Batching**: Random mini-batches of input and target sequences are extracted from the dataset for efficient training.
- **Optimizer**: The `Adam` optimizer is employed to update the model's parameters.
- **Loss Function**: `Cross-entropy loss` quantifies the difference between the model's predicted probabilities and the actual next characters.
- **Regularization**: The training loop incorporates **checkpointing** (saving the model with the best validation performance) and **early stopping** (terminating training if validation loss does not significantly improve over several checks) to prevent overfitting and optimize resource usage.

### **Mini-Batch Stochastic Gradient Descent (SGD)**

Mini-batch SGD is a training optimization method where the model's parameters are updated using a small, randomly selected subset (a "mini-batch") of the training data in each iteration, rather than the entire dataset.

- **Why it's okay**: Using mini-batches is computationally more efficient than processing the entire dataset at once, especially for large datasets. It also introduces some noise into the gradient estimation, which can help the model escape shallow local minima and generalize better.
- **In this architecture**: For each step within the training loop, one mini-batch of data is fed to the model. The gradients are computed and the model's weights are updated based on this single batch, and this process repeats for a specified number of steps or until convergence.
- **Loss Curve Behavior**: Due to the varying gradients from different mini-batches, the training loss curve often appears "zig-zag" or noisy rather than perfectly smooth.

### **Code Structure**

- **`PreprocessingTraining` Class**:

  - Manages data loading, tokenization (converting characters to numerical IDs and vice-versa), and splitting the dataset into training, validation, and test sets.
  - Provides functionality to generate random mini-batches for model input.

- **`TransformerModel` Class**:

  - Defines the neural network architecture, including embedding layers, Q/K/V linear transformations, and the final projection layer.
  - The `forward` method computes logits and loss.
  - The `generate` method enables autoregressive text generation.
  - The `evaluate_validation_loss` method assesses performance on the validation set.
  - The `train_loop` method orchestrates the entire training process, including optimization, logging, checkpointing, early stopping, and plotting loss curves.

- **`hyperparameter_search` Function**:
  - An optional utility to systematically evaluate different combinations of learning rates, batch sizes, and context window lengths to find optimal configurations for the model.
