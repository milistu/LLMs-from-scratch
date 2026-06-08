## Inputs and targets

When building the dataset, for each text chunk we want to have inputs and targets. Even though regular Deep Learning may steer us to think that inputs are a group of features (sequence of tokens) and the target is a single label (single next token in the sequence), that intuition is wrong. Here inputs are a sequence of tokens and targets are **the inputs shifted by one position to the right**. Because of **causal masking**, the attention mechanism internally sees all the (context -> next token) pairs from a single input/target sequence, so we do not need to build them inside the dataset class manually.
```
Input sequence: [1, 2, 3, ..., n]
Target sequence: [2, 3, 4, ..., n + 1]
Pairs the model effectively sees:
1       -> 2
1, 2    -> 3
1, 2, 3 -> 4
...
```

## Stride and the sliding window

We use stride in data sampling to avoid overlaps between inputs, which would essentially show very similar text otherwise. The stride is how far the window moves between samples, so when the **stride equals the window size we get no overlap at all**. Overlapping samples can lead to potential **overfitting** because we are essentially repeating the same input multiple times, and additionally this can reduce the training time and cost.

## Token embeddings

Embeddings are **part of the LLM** and they are **learned during training**, just like any other model parameter.
The conversion from token IDs to embeddings using an embedding layer is essentially a **lookup operation** that retrieves rows from the embedding layer's weight matrix via the token ID.

## Positional embeddings

Because the self-attention mechanism of LLMs is **position-agnostic**, it is helpful to inject additional position information, otherwise we would get the same embedding for a token found in different positions in the sequence.
There are two broad categories of position-aware embeddings:
- **Absolute positional embeddings**: For each position in the input sequence, a unique embedding is added to the token's embedding to convey its exact location.
- **Relative positional embeddings**: Distance between tokens where the model learns the relationships in terms of "how far apart" rather than "at which exact position". The advantage here is that the model can generalize better to sequences of varying lengths, even if it hasn't seen such lengths during training.
OpenAI's GPT models use **absolute** positional embeddings that are **optimised during training** rather than being fixed. This optimisation process is part of the model training itself.
