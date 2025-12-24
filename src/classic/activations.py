import numpy as np


def sigmoid(x):
    """
    Sigmoid activation:
        sigma(x) = 1 / (1 + exp(-x))

    Used in:
    - linear baseline output neuron
    - MLP hidden layer
    - MLP output neuron
    """
    return 1.0 / (1.0 + np.exp(-x))
