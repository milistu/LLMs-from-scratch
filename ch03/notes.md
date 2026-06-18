# Coding attention mechanisms

## The "self" in self-attention

In self-attention, the "self" refers to the mechanism's ability to compute attention weights by relating different positions within a single input sequence. This is in contrast to traditional attention mechanisms, where the focus is on the relationships between elements of two different sequences, such as between an input and an output sequence.

So regular attention builds relationships between two (input and output) sequences, like in the task of translation, whereas self-attention builds relationships between elements of one, single input sequence.

The attention mechanism takes as input a list of token embeddings (input_size × hidden_dim) and outputs the same size — the embedding values are just reconsidered in context. Every token has its own embedding representation, but each token can appear in different settings and different contexts. A single embedding of hidden_size is not enough to capture all the contexts where that token might be found. That's why we have the self-attention mechanism: it takes the full sequence as input, reevaluates each token against all other tokens in the sequence, and gives us the same-size output but with richer contextual information.

## Simple self-attention without trainable weights

The dot product is essentially a similarity measure between two vectors (embeddings). It tells us how aligned two vectors are, which helps attention understand which input tokens have stronger relationships than others. Cosine similarity is similar — it's the dot product divided by the product of the vector magnitudes (i.e., normalized). In attention, we achieve a related kind of normalization when we convert raw attention scores to attention weights via softmax.

## Self-attention with trainable weights

When moving from the simplified version of self-attention — where we use token embeddings directly to calculate attention scores with a dot product — to a mechanism that can learn, we essentially transform each token embedding into three parts. We use trainable weight matrices $W_q$, $W_k$, and $W_v$ to split the token embedding into query, key, and value vectors that are later used for calculating the context vector. This reminds me of the divide and conquer approach: we split (transform) one complex task into multiple focused parts that together can better understand complex patterns.

This mechanism is also called scaled dot-product attention. The "scaled" part comes from dividing the attention scores by $\sqrt{d_k}$ (the square root of the key dimension) before softmax — this prevents the dot products from growing too large in magnitude, which would push the softmax into regions of extremely small gradients.

### Why query, key, value

These terms are borrowed from the domain of information retrieval. We have a query, which is our current token that we focus on. We use that query to search through all tokens in the input sequence and compare it with all keys — similar to searching a database, where keys are used for indexing. Then, based on the relevance (here, the attention weights we calculate), we fetch the values (or meanings) of the tokens, scale them accordingly, and build a unified context vector.

## Causal attention

Causal self-attention blocks future tokens from entering the context of the current token, making sure that only previous and current tokens influence the context vector calculation.

## Multi-head attention

Multi-head attention is essentially multiple instances of the self-attention mechanism, each with its own learned linear projections, running in parallel and then combining their outputs. This is computationally intensive, but it is crucial for the kind of complex pattern recognition that transformer-based LLMs are known for. To put it in perspective: the original transformer paper stacked the encoder-decoder block only six times, while GPT-2 small uses 12 transformer layers (which helped enable its capabilities), and GPT-3 has 96 transformer layers.

The key idea: we run the attention mechanism multiple times in parallel, with different learned linear projections — the results of multiplying the input embeddings by separate weight matrices for each head — so the model can attend to different representation subspaces at different positions simultaneously.

Masked multi-head attention is another name for multi-head attention that uses causal attention.

## PyTorch buffers

When our module contains tensors that aren't trainable weights (parameters) — like a causal mask — we register them as PyTorch buffers instead of keeping them as regular tensor attributes. Buffers solve two problems: they are automatically moved to the correct device when the module is moved (no manual `.to(device)` needed), and they are included in the `state_dict` so they get saved and loaded alongside the model's weights.