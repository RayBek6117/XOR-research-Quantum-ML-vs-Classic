import numpy as np
from .activations import sigmoid


def mlp_forward_proba(x, params):
    """
    1-hidden-layer MLP:
      input (2) -> hidden (n_hidden, sigmoid) -> output (1, sigmoid)

    params:
      W_hidden: (n_hidden, 3)   # 2 inputs + bias
      W_out:    (n_hidden + 1,) # hidden + bias
    """
    W_hidden = params["W_hidden"]
    W_out = params["W_out"]

    x_ext = np.array([x[0], x[1], 1.0], dtype=float)

    h_raw = np.dot(W_hidden, x_ext)
    h = sigmoid(h_raw)
    h_ext = np.append(h, 1.0)

    z_out = np.dot(W_out, h_ext)
    return sigmoid(z_out)


def train_mlp(X, y, n_hidden: int = 2, epochs: int = 5000, lr: float = 0.3, seed: int = 0, verbose_every: int = 1000):
    """
    Train MLP with SGD minimizing MSE (kept exactly like your original logic).
    """
    rng = np.random.default_rng(seed)
    W_hidden = rng.normal(scale=0.5, size=(n_hidden, 3))
    W_out = rng.normal(scale=0.5, size=(n_hidden + 1,))
    n = len(X)

    for epoch in range(epochs):
        loss_sum = 0.0
        idx = rng.permutation(n)

        for i in idx:
            x = X[i]
            t = float(y[i])

            # Forward
            x_ext = np.array([x[0], x[1], 1.0], dtype=float)
            h_raw = np.dot(W_hidden, x_ext)
            h = sigmoid(h_raw)
            h_ext = np.append(h, 1.0)
            z_out = np.dot(W_out, h_ext)
            out = sigmoid(z_out)

            # MSE
            loss = 0.5 * (out - t) ** 2
            loss_sum += loss

            # Backprop
            delta_out = (out - t) * out * (1 - out)
            dW_out = delta_out * h_ext

            delta_hidden = W_out[:-1] * delta_out * h * (1 - h)
            dW_hidden = delta_hidden[:, None] * x_ext[None, :]

            # Update
            W_out -= lr * dW_out
            W_hidden -= lr * dW_hidden

        if verbose_every and epoch % verbose_every == 0:
            print(f"[MLP nh={n_hidden}] Epoch {epoch}, mean MSE = {loss_sum / n:.6f}")

    return {"W_hidden": W_hidden, "W_out": W_out}


def train_mlp_bce_with_curve(
    X, y, n_hidden: int = 4, epochs: int = 3000, lr: float = 0.3, seed: int = 0, record_every: int = 10
):
    """
    Special variant used in your big main():
    - trains MLP using BCE-friendly delta (delta_out = yhat - t)
    - records (epoch, BCE) curve every record_every epochs

    Returns
    -------
    params, curve
      curve: list of (epoch, bce_loss)
    """
    rng = np.random.default_rng(seed)
    W_hidden = rng.normal(scale=0.5, size=(n_hidden, 3))
    W_out = rng.normal(scale=0.5, size=(n_hidden + 1,))
    n = len(X)

    def forward(x):
        x_ext = np.array([x[0], x[1], 1.0], dtype=float)
        h_raw = np.dot(W_hidden, x_ext)
        h = sigmoid(h_raw)
        h_ext = np.append(h, 1.0)
        z = np.dot(W_out, h_ext)
        yhat = sigmoid(z)
        return x_ext, h, h_ext, yhat

    def bce_mean():
        eps = 1e-9
        loss_sum = 0.0
        for x, t in zip(X, y):
            _, _, _, p = forward(x)
            p = float(np.clip(float(p), eps, 1 - eps))
            t = float(t)
            loss_sum += -(t * np.log(p) + (1 - t) * np.log(1 - p))
        return loss_sum / len(X)

    curve = []

    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in idx:
            x = X[i]
            t = float(y[i])
            x_ext, h, h_ext, yhat = forward(x)

            # For sigmoid + BCE: derivative wrt z is (yhat - t)
            delta_out = (yhat - t)
            dW_out = delta_out * h_ext

            delta_hidden = W_out[:-1] * delta_out * h * (1 - h)
            dW_hidden = delta_hidden[:, None] * x_ext[None, :]

            W_out -= lr * dW_out
            W_hidden -= lr * dW_hidden

        if ep % record_every == 0:
            curve.append((ep, bce_mean()))

    return {"W_hidden": W_hidden, "W_out": W_out}, curve
