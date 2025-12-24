import numpy as np


def bce_loss_scalar(p: float, y_true: int, eps: float = 1e-9) -> float:
    """
    Binary cross-entropy for a single probability prediction.
    Clipping avoids log(0).
    """
    p = float(np.clip(float(p), eps, 1.0 - eps))
    t = float(y_true)
    return -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))


def evaluate_binary_mse(predict_proba_fn, params, X, y):
    """
    Evaluate a binary classifier using:
    - accuracy (threshold 0.5)
    - mean MSE: 0.5*(p - y)^2
    """
    n = len(X)
    correct = 0
    loss_sum = 0.0

    for x, t in zip(X, y):
        p = float(predict_proba_fn(x, params))
        pred = 1 if p > 0.5 else 0
        correct += int(pred == int(t))
        loss_sum += 0.5 * (p - float(t)) ** 2

    return correct / n, loss_sum / n


def evaluate_binary_bce(predict_proba_fn, params, X, y):
    """
    Evaluate a binary classifier using:
    - accuracy (threshold 0.5)
    - mean BCE
    """
    n = len(X)
    correct = 0
    loss_sum = 0.0

    for x, t in zip(X, y):
        p = float(predict_proba_fn(x, params))
        pred = 1 if p > 0.5 else 0
        correct += int(pred == int(t))
        loss_sum += bce_loss_scalar(p, int(t))

    return correct / n, loss_sum / n
