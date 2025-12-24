import numpy as np


def make_noisy_xor(n_per_cluster: int, noise: float = 0.10, seed: int | None = None):
    """
    Generate a 2D noisy XOR dataset using 4 Gaussian clusters.

    Centers:
      (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1

    Returns
    -------
    X : np.ndarray of shape (4*n_per_cluster, 2)
    y : np.ndarray of shape (4*n_per_cluster,)
    """

    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [0.0, 0.0],  # class 0
            [1.0, 1.0],  # class 0
            [0.0, 1.0],  # class 1
            [1.0, 0.0],  # class 1
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1], dtype=int)

    X_parts = []
    y_parts = []
    for c, lab in zip(centers, labels):
        pts = c + rng.normal(0.0, noise, size=(n_per_cluster, 2))
        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, lab, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int | None = None,
):

    """
    Stratified train/test split to keep class balance.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """

    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_train = int(train_ratio * len(idx0))
    n1_train = int(train_ratio * len(idx1))

    train_idx = np.concatenate([idx0[:n0_train], idx1[:n1_train]])
    test_idx = np.concatenate([idx0[n0_train:], idx1[n1_train:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
