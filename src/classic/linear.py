import numpy as np
from .activations import sigmoid


def linear_forward_proba(x, params):
    """
    Linear baseline model:
      x in R^2 -> add bias -> z = w · [x1, x2, 1] -> p = sigmoid(z)
    """
    w = params["w"]  # shape (3,)
    x_ext = np.array([x[0], x[1], 1.0], dtype=float)
    z = np.dot(w, x_ext)
    return sigmoid(z)


def train_linear_classifier(X, y, epochs: int = 2000, lr: float = 0.1, seed: int = 0, verbose_every: int = 500):
    """
    Train the linear classifier with SGD minimizing MSE.

    Note: XOR is not linearly separable, so this is a baseline that should fail on clean XOR.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.5, size=3)
    n = len(X)

    for epoch in range(epochs):
        loss_sum = 0.0
        idx = rng.permutation(n)

        for i in idx:
            x = X[i]
            t = float(y[i])

            x_ext = np.array([x[0], x[1], 1.0], dtype=float)
            z = np.dot(w, x_ext)
            out = sigmoid(z)

            # MSE: 0.5*(out - t)^2
            loss = 0.5 * (out - t) ** 2
            loss_sum += loss

            # Backprop for single neuron
            delta = (out - t) * out * (1 - out)
            grad_w = delta * x_ext

            w -= lr * grad_w

        if verbose_every and epoch % verbose_every == 0:
            print(f"[LIN] Epoch {epoch}, mean MSE = {loss_sum / n:.6f}")

    return {"w": w}
