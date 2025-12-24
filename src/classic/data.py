import numpy as np


def train_test_split(X, y, train_ratio: float = 0.8, seed: int = 0):
    """
    Simple train/test split without sklearn.

    Parameters
    ----------
    X : np.ndarray (N, d)
    y : np.ndarray (N,)
    train_ratio : float
    seed : int

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    train_size = int(train_ratio * n)

    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def make_continuous_xor(n_samples: int = 1000, seed: int = 0):
    """
    Continuous XOR (auxiliary dataset).

    Generate X ~ Uniform([0,1]^2).
    Create labels via threshold at 0.5:
      y = XOR( x1>0.5, x2>0.5 )
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    bits = (X > 0.5).astype(int)
    y = (bits[:, 0] ^ bits[:, 1]).astype(int)
    return X, y


def make_noisy_xor(n_per_cluster: int = 1000, noise: float = 0.1, seed: int = 0):
    """
    Noisy cluster XOR (main dataset).

    Four Gaussian clusters centered at:
      (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    """
    rng = np.random.default_rng(seed)

    centers = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0], dtype=int)

    X_list, y_list = [], []
    for c, lab in zip(centers, labels):
        pts = c + noise * rng.normal(size=(n_per_cluster, 2))
        X_list.append(pts)
        y_list.append(np.full(n_per_cluster, lab, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y
