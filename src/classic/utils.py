import os
import csv


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_csv_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Save a list of dicts as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_params_linear(params: dict) -> int:
    """Number of parameters for linear model."""
    return int(params["w"].size)


def count_params_mlp(params: dict) -> int:
    """Number of parameters for 1-hidden-layer MLP."""
    return int(params["W_hidden"].size + params["W_out"].size)
