# PROJECT.md

## Overview
This document describes the functionality of the provided C code (`tf.c`), which implements a transformer model. The code is designed to process tokenized input data and perform both training and inference operations.

The transformer architecture follows the standard design described in "Attention Is All You Need" [1], with multiple attention layers, multi-layer perceptron (MLP) layers, and residual connections. The implementation includes:

- **Token embeddings**: Transforming input tokens into vector representations.
- **Position embeddings**: Adding position information to the token vectors.
- **Multi-head attention**: Computing self-attention over the sequence of token vectors.
- **MLP feed-forward networks**: Processing the output of the attention layer.
- **Residual connections**: Adding the output of each sub-layer to its input.

The code also includes functionality for both training (validation) and inference modes, with appropriate assertions and memory management.

---

## Initialization

### Parameters
The parameters of the transformer model are initialized from a JSON file. Each parameter is stored as a key-value pair in the JSON file, where the value is the corresponding weight or bias vector. The code reads these parameters and maps them to their respective positions in memory using the `tf_offset_size` function.

Example parameters include:
- **Token embeddings**: `wte.weight`
- **Position embeddings**: `wpe.weight`
- **Layer normalization weights**: `ln_1.weight`, `ln_2.weight`
- **Attention layer weights**: `attn.c_attn.weight`, `attn.c_proj.weight`
- **MLP layer weights**: `mlp.c_fc.weight`, `mlp.c_proj.weight`

---

## Input Processing

### Token Embeddings
The input tokens are converted into vector representations using the token embedding matrix (`wte`). Each token is mapped to a fixed-dimensional vector (d_model).

Example:
input_token = decoder->items[input[i]].offset activations[i] += wte[batch][i] * wte.weight[input_token]


### Position Embeddings
Position embeddings are added to the token embeddings to provide positional information. These embeddings are learned during training.

Example:


### Input Normalization
The input sequence is normalized before being processed by the first layer of the transformer.

---

## Model Architecture

### Multi-head Attention
Each attention head computes the following steps:

1. **Query, Key, Value Vectors**:
   - Q = `activations * c_attn.weight`
   - K = `activations * c_attn.weight`
   - V = `activations * c_attn.weight`

2. **Dot Product Attention**:
   - Compute attention scores: \( \text{scores} = \frac{\text{Q} \cdot \text{K}^T}{\sqrt{d_k}} \)
   - Apply softmax to get attention weights.
   - Weighted sum of value vectors.

3. **Projection**:
   - Transform the concatenated attention output back to d_model dimensions: \( \text{output} = \text{V} \cdot c\_proj.weight \)

### MLP Layer
Each MLP layer consists of two fully connected layers with GELU activation:

1. First projection: \( \text{input} \rightarrow 4d_\text{model} \)
2. GELU non-linearity.
3. Second projection: \( 4d_\text{model} \rightarrow d_\text{model} \)

---

## Training vs Inference

### Training (Validation Mode)
- The model processes input sequences and computes gradients using backpropagation.
- Activations are randomly initialized for validation.
- Gradients are computed using the `tf_process` function.

Example usage:


### Inference Mode
- The model generates output tokens one by one.
- For each step:
  - Process the current input sequence using `tf_process`.
  - Write the generated token to standard output.
  - Append the generated token to the input sequence for the next step.

Example usage:


---

## Output Processing

### Final Layer Normalization
The output of the transformer is normalized using a final layer normalization.

Example:


### Token Generation
The normalized output is projected back to the vocabulary space using the token embedding weights (`wte`).

Example:


---

## Loss Calculation

The code uses cross-entropy loss for training. For each position in the sequence, the loss is computed as:

\[
\text{loss} = -\log(p_{\text{correct}})
\]

where \( p_{\text{correct}} \) is the predicted probability of the correct token.

---

## Notes

1. **Matrix Dimensions**:
   - d_seq: Sequence dimension.
   - d_model: Model dimension (embedding size).
   - n_heads: Number of attention heads.
   - n_layers: Number of transformer layers.
   - d_k: Dimension per attention head (\( d_{\text{model}} / n_{\text{heads}} \)).

2. **Implementation Details**:
   - The code uses row-major order for matrix multiplication to optimize cache locality.
   - Backpropagation through the embedding layer is implemented carefully to avoid cache misses.

---

## References

1. "Attention Is All You Need" [Original Paper](https://arxiv.org/abs/1706.03798)

This document provides a high-level overview of the transformer implementation in `tf.c`. For detailed information, refer to the source code and comments.

