# **Introduction**

This repository contains a step-by-step implementation of a decoder-only Transformer model trained to predict the next token in a sequence. It is inspired by Andrej Karpathy's "Let's build GPT from scratch" tutorial, but significantly extended with additional features and improvements.

The code progresses from a basic bigram language model to a full multi-head, multi-layer Transformer architecture.

### **Key Enhancements**

- **Bigram and Transformer Models**: Includes implementations of a bigram model, a single-head single-layer Transformer, and a multi-head multi-layer Transformer.
- **Token Granularity**: Supports both character-level and word-level tokenization, allowing experimentation with different input granularities.
- **Training Features**: Adds evaluation metrics, checkpointing, and early stopping to make training more robust.
- **Text Generation Improvements**: Implements generation controls like temperature and top-k sampling to allow more flexible output generation.
- **Hyperparameter Tuning**: Enables configurable search over batch size, learning rate, context window, and other parameters to optimize training.
