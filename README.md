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

Mini-batch SGD is a training optimization method where the model's parameters are updated using a small, randomly selected subset (a "mini-batch") of the training data in each iteration/step, rather than the entire dataset.

- **Why it's okay**: Using mini-batches is computationally more efficient than processing the entire dataset at once, especially for large datasets. It also introduces some noise into the gradient estimation, which can help the model escape shallow local minima and generalize better.
- **In this architecture**: For each step within the training loop, one mini-batch of data is fed to the model. The gradients are computed and the model's weights are updated based on this single batch, and this process repeats for a specified number of steps or until convergence.
- **Loss Curve Behavior**: Due to the varying gradients from different mini-batches, the training loss curve often appears "zig-zag" or noisy rather than perfectly smooth.
- **Training Step/Iteration**: One single update of the model's parameters.
- **Epoch**: An epoch is defined as one complete pass through the entire training dataset. If your training dataset has N samples and your batch size is B, then one epoch consists of N / B training steps typically.
- **Random Sampling Effect**: When get_batch_indices uses torch.randint to pick random starting points for your sequences, it's entirely possible for some sequences to be picked multiple times within what would conceptually be one epoch, while others might not be picked at all. This is the nature of random sampling with replacement. However random sampling allows for continuous training without needing to pre-process or shuffle the entire dataset for each "epoch." Over many thousands or millions of training steps, even with random sampling, it's highly probable that most, if not all, of the training data will be exposed to the model multiple times.

### **Inference (Text Generation) and Context Length**

During text generation, how the model handles the input prompt's length is crucial:

- **Our Implementation's Approach**: In this specific implementation, the `generate` method is designed to always process a fixed context length. If a user provides a prompt longer than the `context_window` (the sequence size used during training batches), the model will **truncate the prompt and only consider the last `context_window` tokens** for its predictions. This means earlier parts of a very long prompt will not directly influence the generated text.

- **General Transformer Models' Approach**: In contrast, many larger Transformer models do not have this strict truncation at every step. Their learnable parameters (weights and biases for linear layers, embeddings, etc.) **do not depend on the sequence length or the batch size** defined during training. The primary constraint on input sequence length for such models comes from their **positional encoding mechanism**. If a model's positional encodings can handle longer sequences, it can process and draw context from an entire long prompt (up to its maximum sequence length) for its initial predictions, then use efficient techniques like key-value caching for subsequent autoregressive generation.

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

## **Multi-Head, Multi-Layer Transformer Language Model**

This repository showcases a more advanced character-level Transformer model, incorporating multiple attention heads and stacked Transformer layers. It demonstrates a scaled-up architecture for improved context understanding and next-character prediction.

### **Model Architecture**

This Transformer model predicts the next character by learning complex dependencies across the input sequence.

- **Embedding Layers**: The model starts by combining `token embeddings` (character-to-vector mapping) and `positional embeddings` (encoding character position) to provide initial rich representations.
- **Multi-Head Attention (MHA)**: Instead of a single attention mechanism, MHA allows the model to jointly attend to information from different representation subspaces at different positions. It concatenates outputs from multiple "heads," each focusing on a different aspect of the input, leading to a richer context.
  - **Shared Linear Projections**: `Query (Q)`, `Key (K)`, and `Value (V)` are created using shared linear layers across all heads for efficiency, and then reshaped to distribute into separate heads.
  - **Scaled Dot-Product Attention**: Each head performs scaled dot-product attention, applying a `causal mask` to ensure predictions are based only on preceding characters.
- **Transformer Block**: Each block consists of a Multi-Head Attention layer followed by a `Feed-Forward Network (FFN)`. Both layers are wrapped with `Layer Normalization` and `Residual Connections` to stabilize training and enable deeper networks.
- **Stacked Transformer Blocks**: The model uses multiple `Transformer Blocks` stacked sequentially, allowing for hierarchical feature extraction and deeper understanding of context.
- **Projection Layer**: A final linear layer projects the output of the Transformer stack to `logits` over the vocabulary, representing the model's predictions for the next character.

### **Training Method**

The model is trained using **mini-batch stochastic gradient descent**.

- **Batching**: Random mini-batches of input and target sequences are used for each training step.
- **Optimizer**: The `Adam` optimizer is employed to update the model's parameters.
- **Loss Function**: `Cross-entropy loss` quantifies the difference between predicted and actual next characters.
- **Regularization**: Training incorporates **checkpointing** (saving the best model based on validation loss) and **early stopping** (halting training if validation loss plateaus) to prevent overfitting and optimize resource usage.
- **Loss Curve Behavior**: Due to the varying gradients from different mini-batches, the training loss curve often appears "zig-zag" or noisy rather than perfectly smooth.

### **Code Structure**

- **`PreprocessingTraining` Class**:

  - Handles dataset loading, character-to-integer mapping (tokenization), and splitting data into training, validation, and test sets.
  - Provides functionality to generate random mini-batches for model input.

- **`MultiHeadAttention` Class**:

  - Defines the multi-head self-attention mechanism, including linear projections for Q, K, V, scaling, causal masking, and combining attention outputs from multiple heads.

- **`TransformerBlock` Class**:

  - Encapsulates a single layer of the Transformer architecture, combining a `MultiHeadAttention` layer and a `Feed-Forward Network` with `Layer Normalization` and `Residual Connections`.

- **`TransformerModel` Class**:

  - Defines the complete Transformer network by composing token and positional embeddings, a stack of `TransformerBlock` instances, and the final output projection.
  - The `forward` method performs the full pass through the network, returning logits and loss.
  - The `generate` method enables autoregressive text generation.
  - The `evaluate_validation_loss` method assesses performance on the validation set.
  - The `train_loop` method orchestrates the entire training process, including optimization, logging, checkpointing, early stopping, and plotting loss curves. This class is responsible for running the training.

- **`hyperparameter_search` Function**:
  - An optional utility to systematically evaluate different combinations of learning rates, batch sizes, and context window lengths to identify optimal training configurations.

## **Multi-Head, Multi-Layer Transformer Language Model (Word-Level)**

This repository presents a multi-head, multi-layer Transformer model designed for next-word prediction. Unlike previous versions that operated on characters, this model processes text at the **word level**, leveraging a pre-trained tokenizer for richer semantic understanding. It incorporates advanced Transformer functionalities for improved context comprehension and generation.

### **Key Changes and Enhancements**

The core Transformer architecture remains, but significant modifications have been made for word-level processing and enhanced training/generation:

- **Word-Level Tokenization**: Instead of character-by-character processing, this model now uses a pre-trained `AutoTokenizer` from the `transformers` library to convert raw text into sequences of word tokens (or subword tokens, depending on the tokenizer). This significantly increases the vocabulary size but allows the model to learn more meaningful linguistic patterns.
- **Enhanced Generation**: The `generate` method has been improved to offer more control over text generation, including:
  - **Top-K Sampling**: Samples the next token only from the `k` most probable tokens, preventing the generation of very unlikely words.
  - **Temperature Scaling**: Adjusts the randomness of the sampling process; higher temperatures lead to more diverse outputs, while lower temperatures make the output more deterministic.
- **Training Loop Metrics**: The `train_loop` and `evaluate_validation_loss` methods now use `train_iters` and `eval_iters` to control the number of batches processed during each training and evaluation phase, respectively, rather than a fixed number of epochs. This provides more granular control over the training process and evaluation frequency.
- **Device Management**: Explicit device (CPU/CUDA) management is included to ensure the model and data are processed on the available hardware.

### **Model Architecture**

This Transformer model predicts the next word by analyzing the context of preceding words.

- **Embedding Layers**: The model starts by combining `token embeddings` (mapping each word token to a learnable vector) and `positional embeddings` (encoding word position) to provide initial rich representations.
- **Multi-Head Attention (MHA)**: Allows the model to jointly attend to information from different representation subspaces at different positions within the sequence. It concatenates outputs from multiple "heads," each focusing on a different aspect of the input context.
  - **Shared Linear Projections**: `Query (Q)`, `Key (K)`, and `Value (V)` are created using shared linear layers for efficiency and then reshaped to distribute into separate heads.
  - **Scaled Dot-Product Attention**: Each head performs scaled dot-product attention, applying a `causal mask` to ensure predictions are based only on preceding words.
- **Transformer Block**: Each block consists of a `MultiHeadAttention` layer followed by a `Feed-Forward Network (FFN)`. Both layers are wrapped with `Layer Normalization` and `Residual Connections` to stabilize training and enable deeper networks.
- **Stacked Transformer Blocks**: The model uses multiple `Transformer Blocks` stacked sequentially, allowing for hierarchical feature extraction and deeper understanding of context.
- **Projection Layer**: A final linear layer projects the output of the Transformer stack to `logits` over the entire word vocabulary, representing the model's predictions for the next word.

### **Training Method**

The model is trained using **mini-batch stochastic gradient descent**.

- **Batching**: Random mini-batches of input and target sequences are used for each training step.
- **Optimizer**: The `Adam` optimizer is employed to update the model's parameters.
- **Loss Function**: `Cross-entropy loss` quantifies the difference between predicted probabilities and actual next words.
- **Regularization**: Training incorporates **checkpointing** (saving the best model based on validation loss) and **early stopping** (halting training if validation loss plateaus) to prevent overfitting and optimize resource usage.
- **Loss Curve Behavior**: Due to the varying gradients from different mini-batches, the training loss curve often appears "zig-zag" or noisy rather than perfectly smooth.

### **Mini-Batch Stochastic Gradient Descent (SGD)**

Mini-batch SGD is a training optimization method where the model's parameters are updated using a small, randomly selected subset (a "mini-batch") of the training data in each iteration, rather than the entire dataset.

- **Why it's okay**: Using mini-batches is computationally more efficient than processing the entire dataset at once, especially for large datasets. It also introduces some noise into the gradient estimation, which can help the model escape shallow local minima and generalize better.
- **In this architecture**: For each step within the training loop, one mini-batch of data is fed to the model. The gradients are computed and the model's weights are updated based on this single batch, and this process repeats for a specified number of steps or until convergence.
- **Loss Curve Behavior**: Due to the varying gradients from different mini-batches, the training loss curve often appears "zig-zag" or noisy rather than perfectly smooth.

### **Code Structure**

- **`PreprocessingTraining` Class**:

  - Handles dataset loading, **word-level tokenization** using `AutoTokenizer`, and splitting data into training, validation, and test sets.
  - Provides functionality to generate random mini-batches for model input.

- **`MultiHeadAttention` Class**:

  - Defines the multi-head self-attention mechanism, including linear projections for Q, K, V, scaling, causal masking, and combining attention outputs from multiple heads.
  - It doesn't directly interact with `PreprocessingTraining` or `TransformerModel` initially, but rather, an instance of `MultiHeadAttention` is _contained within_ each `TransformerBlock`.

- **`TransformerBlock` Class**:

  - Encapsulates a single layer of the Transformer architecture, combining a `MultiHeadAttention` layer and a `Feed-Forward Network` with `Layer Normalization` and `Residual Connections`.
  - Each `TransformerBlock` instance _uses_ a `MultiHeadAttention` instance internally, along with a feed-forward network, layer normalization, and residual connections. The `TransformerModel` then stacks multiple `TransformerBlock` instances.

- **`TransformerModel` Class**:

  - Defines the complete Transformer network by composing token and positional embeddings, a stack of `TransformerBlock` instances, and the final output projection.
  - The `forward` method performs the full pass through the network, returning logits and loss. During its `forward` pass, it sequentially _calls_ the `forward` method of each `TransformerBlock` to process the input.
  - The `generate` method enables autoregressive text generation with `top-k sampling` and `temperature scaling`.
  - The `evaluate_validation_loss` method assesses performance on the validation set using `eval_iters`.
  - The `train_loop` method orchestrates the entire training process, including optimization, logging, checkpointing, early stopping, and plotting loss curves, running for `train_iters`. This class is responsible for running the training.

- **`hyperparameter_search` Function**:
  - An optional utility to systematically evaluate different combinations of learning rates, batch sizes, and context window lengths to find optimal training configurations.
