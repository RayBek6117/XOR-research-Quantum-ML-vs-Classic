import numpy as np


def fit_minmax_scaler(X_train: np.ndarray):

    """
    Fit a simple min-max scaler on TRAIN set only.

    Returns
    -------
    mins : np.ndarray shape (2,)
    span : np.ndarray shape (2,)
        span is forced to be non-zero to avoid division by zero.
    """

    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    span = maxs - mins
    span = np.where(span < 1e-12, 1.0, span)
    return mins, span


def to_angles(X: np.ndarray, mins: np.ndarray, span: np.ndarray):

    """
    Map features to angles in [0, pi] using min-max scaling.

    This often improves decision boundary visibility:
    raw XOR coordinates are near {0,1}, and without scaling the circuit can
    collapse to a nearly constant output.

    Returns
    -------
    X_angles : np.ndarray, same shape as X
    """

    return np.pi * (X - mins) / span
