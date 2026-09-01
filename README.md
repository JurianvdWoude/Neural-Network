
# Neural Network from Scratch

A fully-connected feedforward neural network implemented from scratch using **Python and NumPy**, without using a machine-learning or deep-learning framework.

I originally built this project as a way to understand how neural networks work under the hood, including forward propagation, backpropagation, gradient descent, activation functions, and regularization.

The network is currently being used for a small **cat vs. non-cat image classification experiment** using 64×64 RGB images.

## Features

* L-layer fully-connected neural network
* ReLU activation for hidden layers
* Sigmoid activation for binary classification
* He weight initialization
* Forward and backward propagation
* Gradient descent
* Binary cross-entropy cost
* L2 regularization
* Dropout regularization
* Separate training, prediction, and evaluation scripts
* Custom image dataset generated with my [catscraper](https://github.com/JurianvdWoude/catscraper) project

## Project Structure

```text
Neural-Network/
├── nn.py             # Neural network implementation
├── data_loader.py    # Loads and prepares image data
├── train.py          # Trains and saves the model
├── predict.py        # Makes predictions on images
├── evaluate.py       # Evaluates the model on test data
├── data/             # Local dataset (not committed to Git)
└── model/            # Saved model parameters
```

The dataset is intentionally not included in this repository. It is generated separately using [catscraper](https://github.com/JurianvdWoude/catscraper).

## The Neural Network

The network follows the standard training process:

```text
Input image
     ↓
Linear layer + ReLU
     ↓
Linear layer + ReLU
     ↓
       ...
     ↓
Linear layer + Sigmoid
     ↓
Cat / Non-cat prediction
```

The hidden layers use **ReLU**, while the final layer uses **Sigmoid** to produce an output between 0 and 1 for binary classification.

The current label convention is:

```text
0 = Cat
1 = Non-cat
```

## Training

The network can be trained using:

```bash
python train.py
```

Training data consists of 64×64 RGB images split into `cat` and `nocat` directories.

The dataset is randomly split into training and test sets by the accompanying `catscraper` project.

## Evaluation

After training, the model can be evaluated on the test set with:

```bash
python evaluate.py
```

The evaluation reports the model's accuracy and a simple confusion matrix.

For example:

```text
Test images: 634
Accuracy: 58.83%

Confusion matrix:
-----------------
Actual cats:
  predicted cat:     263
  predicted non-cat: 122

Actual non-cats:
  predicted cat:     139
  predicted non-cat: 110
```

The current accuracy is intentionally included here as a snapshot of the project's current state rather than presenting the project as a highly accurate image classifier. The main purpose of the project is to demonstrate an understanding of how a neural network can be implemented and trained from the ground up.

## Why I Built This

Rather than using an existing machine-learning framework, I wanted to understand what actually happens inside a neural network.

The project therefore implements the core operations manually, including:

* matrix-based forward propagation
* activation functions
* cost calculation
* backpropagation and the chain rule
* gradient descent
* weight initialization
* dropout
* L2 regularization

This project is primarily a learning and hobby project rather than an attempt to build a production-quality image classifier.

## Requirements

* Python 3.x
* NumPy
* Pillow

Install the dependencies with:

```bash
pip install numpy pillow
```

## Related Project

The image dataset used for the experiment is generated with:

**[catscraper](https://github.com/JurianvdWoude/catscraper)**

It downloads images and randomly splits them into training and test datasets. The generated images are kept locally and are not committed to this repository.

## License

This project is open source. See the repository for details.

