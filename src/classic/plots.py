import numpy as np
import matplotlib.pyplot as plt


def plot_dataset_scatter(X, y, title: str, save_path: str):
    """Scatter plot for XOR dataset."""
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], edgecolor="k", color="white", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], edgecolor="k", color="black", label="Class 1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_decision_boundary(predict_proba_fn, params, X, y, title: str, save_path: str, grid_size: int = 250, pad: float = 0.5):
    """
    Plot decision boundary contour for p(class=1)=0.5.
    Uses a flattened grid for speed (no nested loops).
    """
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.zeros(grid_points.shape[0], dtype=float)

    for k, pt in enumerate(grid_points):
        Z[k] = float(predict_proba_fn(pt, params))

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], edgecolor="k", color="white", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], edgecolor="k", color="black", label="Class 1")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_curve(x_vals, y_vals_dict: dict[str, list[float]], title: str, xlabel: str, ylabel: str, save_path: str):
    """Generic curve plot supporting multiple labeled lines."""
    plt.figure(figsize=(7, 5))
    for name, ys in y_vals_dict.items():
        plt.plot(x_vals, ys, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
