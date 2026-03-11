import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x_new = np.array(x)
    return np.where(x_new < 0, x_new*alpha, x_new)