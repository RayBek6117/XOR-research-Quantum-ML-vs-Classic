"""
Required classic experiments for the XOR study.

This script generates the *mandatory* figures and tables for the
classical baselines, exactly as used in the main report:

1) Dataset B scatter plot (sigma=0.10, n_per_cluster=100)
2) Decision boundaries:
   - Linear model
   - MLP (h=4)
3) Learning curve (loss vs epoch):
   - MLP (h=4)
4) Hidden units study for MLP (h in [1,2,4,8])

All outputs are saved to:
  results/classic_required/
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.classic.utils import ensure_dir, save_csv_rows, count_params_mlp
from src.classic.data import make_noisy_xor, train_test_split
from src.classic.metrics import evaluate_binary_bce
from src.classic.linear import train_linear_classifier, linear_forward_proba
from src.classic.mlp import (
    train_mlp,
    train_mlp_bce_with_curve,
    mlp_forward_proba,
)
from src.classic.plots import (
    plot_dataset_scatter,
    plot_decision_boundary,
)


def main():
    # -------------------------------------------------
    # IO
    # -------------------------------------------------
    OUT_DIR = "results/classic_required"
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    TAB_DIR = os.path.join(OUT_DIR, "tables")
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(TAB_DIR)

    # -------------------------------------------------
    # Fixed experimental setting (Dataset B)
    # -------------------------------------------------
    sigma = 0.10
    n_per_cluster = 100
    data_seed = 0
    split_seed = 42

    # -------------------------------------------------
    # Generate dataset
    # -------------------------------------------------
    X, y = make_noisy_xor(
        n_per_cluster=n_per_cluster,
        noise=sigma,
        seed=data_seed,
    )

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, train_ratio=0.8, seed=split_seed
    )

    # -------------------------------------------------
    # 1) Scatter plot (required)
    # -------------------------------------------------
    plot_dataset_scatter(
        X, y,
        title="Dataset B (Noisy Cluster XOR) | sigma=0.10, n_per_cluster=100",
        save_path=os.path.join(FIG_DIR, "datasetB_scatter_sigma0.10_n100.png"),
    )

    # -------------------------------------------------
    # 2) Decision boundary — Linear
    # -------------------------------------------------
    lin_params = train_linear_classifier(
        X_train, y_train,
        epochs=2000,
        lr=0.1,
        seed=0,
        verbose_every=0,
    )

    plot_decision_boundary(
        linear_forward_proba,
        lin_params,
        X, y,
        title="Decision Boundary | Linear | Dataset B | sigma=0.10, n=100",
        save_path=os.path.join(
            FIG_DIR,
            "boundary_linear_B_sigma0.10_n100.png",
        ),
    )

    # -------------------------------------------------
    # 3) Decision boundary — MLP (h=4)
    # -------------------------------------------------
    mlp_params = train_mlp(
        X_train, y_train,
        n_hidden=4,
        epochs=3000,
        lr=0.3,
        seed=0,
        verbose_every=0,
    )

    plot_decision_boundary(
        mlp_forward_proba,
        mlp_params,
        X, y,
        title="Decision Boundary | MLP (h=4) | Dataset B | sigma=0.10, n=100",
        save_path=os.path.join(
            FIG_DIR,
            "boundary_mlp_h4_B_sigma0.10_n100.png",
        ),
    )

    # -------------------------------------------------
    # 4) Learning curve — MLP (h=4, BCE)
    # -------------------------------------------------
    mlp_curve_params, curve = train_mlp_bce_with_curve(
        X_train,
        y_train,
        n_hidden=4,
        epochs=3000,
        lr=0.3,
        seed=0,
        record_every=10,
    )

    epochs = [ep for ep, _ in curve]
    losses = [loss for _, loss in curve]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, losses, label="MLP (h=4) train BCE")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("Learning Curve | MLP (h=4) | Dataset B (sigma=0.10, n=100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIG_DIR,
            "learning_curve_mlp_h4_B_sigma0.10_n100.png",
        ),
        dpi=200,
    )
    plt.close()

    # -------------------------------------------------
    # 5) Hidden units study — MLP
    # -------------------------------------------------
    hidden_list = [1, 2, 4, 8]
    seeds_model = [0, 1, 2, 3, 4]

    rows = []

    for nh in hidden_list:
        acc_vals = []
        times = []

        for sm in seeds_model:
            t0 = time.perf_counter()
            params = train_mlp(
                X_train, y_train,
                n_hidden=nh,
                epochs=3000,
                lr=0.3,
                seed=sm,
                verbose_every=0,
            )
            times.append(time.perf_counter() - t0)

            acc, _ = evaluate_binary_bce(
                mlp_forward_proba,
                params,
                X_test,
                y_test,
            )
            acc_vals.append(acc)

        acc_vals = np.array(acc_vals)
        times = np.array(times)

        rows.append({
            "n_hidden": nh,
            "n_params": count_params_mlp(params),
            "test_acc_mean": float(acc_vals.mean()),
            "test_acc_std": float(acc_vals.std(ddof=1)),
            "train_time_mean_sec": float(times.mean()),
            "train_time_std_sec": float(times.std(ddof=1)),
        })

    save_csv_rows(
        os.path.join(
            TAB_DIR,
            "table_mlp_hidden_units_study_B_sigma0.10_n100.csv",
        ),
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )

    print("Classic REQUIRED experiments saved to:")
    print(" -", FIG_DIR)
    print(" -", TAB_DIR)


if __name__ == "__main__":
    main()
