# Neural Network from Scratch

A fully-connected feedforward neural network implemented from scratch using **NumPy** — no deep learning frameworks required. Supports arbitrary depth, two activation functions, dropout regularization, and L2 regularization.

---

## Features

- **L-layer deep neural network** — configure any number of hidden layers via a single list
- **Activation functions** — ReLU (hidden layers) and Sigmoid (output layer)
- **Two weight initialization strategies:**
  - Standard random initialization (scaled by 0.01)
  - He initialization (recommended for ReLU networks)
- **Forward propagation** with and without dropout
- **Backward propagation** with:
  - Standard backprop
  - L2 regularization
  - Dropout regularization
- **Cross-entropy cost** function
- **Gradient descent** parameter updates
- **Dropout** to prevent exploding/vanishing gradients and overfitting

---

## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)

Install NumPy with:

```bash
pip install numpy
```

---

## File Structure

```
Neural-Network/
├── nn.py        # Full neural network implementation
└── README.md
```

---

## Usage

### 1. Initialize parameters

```python
from nn import initialize_parameters_deep, he_initialize_parameters_deep

# Standard initialization
parameters = initialize_parameters_deep([12288, 20, 7, 5, 1])

# He initialization (better for deep ReLU networks)
parameters = he_initialize_parameters_deep([12288, 20, 7, 5, 1])
```

`layer_dims` is a list where each element defines the number of units in that layer (input → hidden → ... → output).

### 2. Train the model

```python
from nn import L_layer_model

parameters, costs = L_layer_model(
    X,                    # Input data, shape (n_x, m)
    Y,                    # Labels, shape (1, m)
    layers_dims,          # e.g. [n_x, 20, 7, 5, 1]
    learning_rate=0.0075,
    num_iterations=3000,
    print_cost=True,
    lambd=0,              # L2 regularization strength (0 = disabled)
    keep_prob=1           # Dropout keep probability (1 = disabled)
)
```

**Regularization options (mutually exclusive):**

| Option | Effect |
|---|---|
| `lambd=0, keep_prob=1` | No regularization (standard backprop) |
| `lambd > 0` | L2 regularization |
| `keep_prob < 1` | Dropout regularization |

### 3. Forward pass only

```python
from nn import L_model_forward, L_model_forward_with_dropout

AL, caches = L_model_forward(X, parameters)

# With dropout (use during training only)
AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob=0.8)
```

---

## API Reference

### Initialization

| Function | Description |
|---|---|
| `initialize_parameters(n_x, n_h, n_y)` | Initialize a 2-layer network |
| `initialize_parameters_deep(layer_dims)` | Initialize an L-layer network |
| `he_initialize_parameters_deep(layer_dims)` | He initialization for deep ReLU networks |

### Forward Propagation

| Function | Description |
|---|---|
| `L_model_forward(X, parameters)` | Standard forward pass |
| `L_model_forward_with_dropout(X, parameters, keep_prob)` | Forward pass with dropout |

### Cost

| Function | Description |
|---|---|
| `compute_cost(AL, Y)` | Cross-entropy cost |
| `compute_cost_with_regularization(AL, Y, parameters, lambd)` | Cross-entropy + L2 cost |

### Backward Propagation

| Function | Description |
|---|---|
| `L_model_backward(AL, Y, caches)` | Standard backprop |
| `L_model_backward_with_regularization(AL, Y, caches, lambd)` | Backprop with L2 regularization |
| `L_model_backward_with_dropout(AL, Y, caches, keep_prob)` | Backprop with dropout |

### Parameter Update

| Function | Description |
|---|---|
| `update_parameters(params, grads, learning_rate)` | Gradient descent update step |

---

## How It Works

The network uses a standard **forward → cost → backward → update** training loop:

1. **Forward propagation** — compute activations layer by layer (ReLU for hidden layers, Sigmoid for the output)
2. **Cost computation** — binary cross-entropy loss, optionally with L2 penalty
3. **Backward propagation** — compute gradients via chain rule, layer by layer
4. **Parameter update** — apply gradient descent using the computed gradients

Dropout is applied during the forward pass by randomly zeroing out neurons, then scaled by `1/keep_prob` to maintain expected activation values. During backprop, the same dropout mask is reapplied.

---

## License

This project is open source. See the repository for details.
