## Normalisation

When training deep neural networks, problems like vanishing or exploding gradients can occur. Vanishing gradients happen when the gradients flowing backward through the network become extremely small, so the weights barely update. Exploding gradients are the opposite — gradients grow so large that weight updates become erratic and unstable. In both cases, the learning process struggles to find good parameters that minimise the loss function.

To mitigate these effects, we use normalisation. The idea is to center the outputs (activations) of a neural network layer around a mean of 0 and normalise their variance to 1, also known as unit variance. By keeping the signal in a stable numerical range, this stabilises training, enables faster convergence to effective weights, and helps gradients flow properly through deep networks.

### Layer normalisation vs. batch normalisation

Unlike batch normalisation, which normalises across the batch dimension, layer normalisation normalises across the feature dimension. With LLMs, compute resources and hardware dictate the batch size during both training and inference. Since layer normalisation normalises each input independently of the batch size, it is more flexible and stable for transformer architectures.

##  Feed forward network & GELU

### GELU activation

GELU (Gaussian Error Linear Unit) is a non-linear activation function used in GPT-2 and many modern transformers. Unlike ReLU, which hard-clips negative values to exactly zero, GELU is smooth — it allows small negative values to pass through, weighted probabilistically. This smoothness helps gradients flow better during training compared to the sharp cutoff of ReLU.

### FeedForward neural network

The feedforward network (FFN) in a transformer block expands the embedding dimension (e.g., from 768 to 3072) and then contracts it back down. This expansion into a higher-dimensional space gives the network room to learn more complex patterns that wouldn't be linearly separable in the original lower-dimensional space.

A good way to picture it: imagine you want to separate a ring of dots from a circle of dots inside it using a single straight line. In 2D, it's impossible. But if you lift the inner circle into a third dimension, you can separate them with a flat plane. That's what the FFN's hidden layer does — it temporarily projects into a higher dimension where patterns become easier to separate, then brings the result back down.


## Shortcut Connections

Shortcut connections are also called skip or residual connections, originally introduced in deep networks in CV (residual networks).
A shortcut connection creates an alternative shorter path for the gradient to flow through the network. This is  achieved by adding the output of one layer to the output of a later layer, skipping one or more layers in between.