import os
import time
import numpy as np

from src.classic.utils import ensure_dir, save_csv_rows
from src.classic.data import make_noisy_xor, train_test_split
from src.classic.metrics import evaluate_binary_mse  # returns (acc, mse)
from src.classic.plots import plot_dataset_scatter, plot_decision_boundary, plot_curve

from src.classic.linear import train_linear_classifier, linear_forward_proba
from src.classic.mlp import train_mlp, mlp_forward_proba

def count_params_linear(params):
    # params["w"] shape = (3,)
    return int(params["w"].size)

def count_params_mlp(params):
    # W_hidden shape = (n_hidden, 3), W_out shape = (n_hidden+1,)
    return int(params["W_hidden"].size + params["W_out"].size)

# -----------------------------
# Helpers
# -----------------------------

def mean_std(values):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def run_one_experiment_B(
    n_per_cluster, sigma, data_seed, split_seed,
    model_name, model_cfg, train_cfg
):
    """
    Single run on Dataset B (noisy cluster XOR) for ONE model.
    Returns:
      row, (X,y), (X_train,y_train,X_test,y_test), params
    """
    # data + split (done ONLY here, no double-generation in main)
    X, y = make_noisy_xor(n_per_cluster=n_per_cluster, noise=sigma, seed=data_seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8, seed=split_seed)

    t0 = time.perf_counter()

    if model_name == "linear":
        params = train_linear_classifier(
            X_train, y_train,
            epochs=train_cfg["epochs"],
            lr=train_cfg["lr"],
            seed=train_cfg["seed_model"],
            verbose_every=0
        )
        pred_fn = linear_forward_proba
        n_params = count_params_linear(params)

    elif model_name == "mlp":
        params = train_mlp(
            X_train, y_train,
            n_hidden=model_cfg["n_hidden"],
            epochs=train_cfg["epochs"],
            lr=train_cfg["lr"],
            seed=train_cfg["seed_model"],
            verbose_every=0
        )
        pred_fn = mlp_forward_proba
        n_params = count_params_mlp(params)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_time = time.perf_counter() - t0

    train_acc, train_loss = evaluate_binary_mse(pred_fn, params, X_train, y_train)
    test_acc, test_loss = evaluate_binary_mse(pred_fn, params, X_test, y_test)

    row = {
        "dataset": "B_noisy_cluster",
        "sigma": float(sigma),
        "n_per_cluster": int(n_per_cluster),
        "n_samples": int(len(X)),
        "data_seed": int(data_seed),
        "split_seed": int(split_seed),

        "model": str(model_name),
        "n_hidden": int(model_cfg.get("n_hidden", -1)),

        "epochs": int(train_cfg["epochs"]),
        "lr": float(train_cfg["lr"]),
        "seed_model": int(train_cfg["seed_model"]),

        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "train_loss_mse": float(train_loss),
        "test_loss_mse": float(test_loss),

        "n_params": int(n_params),
        "train_time_sec": float(train_time),
    }

    return row, (X, y), (X_train, y_train, X_test, y_test), params


def train_mlp_with_curve_mse(
    X, y,
    n_hidden=4,
    epochs=3000,
    lr=0.3,
    seed=0,
    record_every=10
):
    """
    Train MLP and record TRAIN MSE every `record_every` epochs.
    This is local to the script so you don't need to modify src.
    Architecture matches your MLP:
      W_hidden: (n_hidden, 3), W_out: (n_hidden + 1,)
    """
    rng = np.random.default_rng(seed)

    # Initialize weights
    W_hidden = rng.normal(scale=0.5, size=(n_hidden, 3))
    W_out = rng.normal(scale=0.5, size=(n_hidden + 1,))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward_one(x):
        x_ext = np.array([x[0], x[1], 1.0], dtype=float)
        h_raw = W_hidden @ x_ext
        h = sigmoid(h_raw)
        h_ext = np.append(h, 1.0)
        z_out = W_out @ h_ext
        yhat = sigmoid(z_out)
        return x_ext, h, h_ext, yhat

    curve_epochs = []
    curve_losses = []

    n = len(X)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in idx:
            x = X[i]
            t = float(y[i])

            x_ext, h, h_ext, yhat = forward_one(x)

            # MSE loss: 0.5*(yhat - t)^2
            # Backprop
            dL_dy = (yhat - t)
            dy_dz = yhat * (1.0 - yhat)
            delta_out = dL_dy * dy_dz

            dW_out = delta_out * h_ext
            delta_h = W_out[:-1] * delta_out * h * (1.0 - h)
            dW_hidden = delta_h[:, None] * x_ext[None, :]

            W_out -= lr * dW_out
            W_hidden -= lr * dW_hidden

        if ep % record_every == 0:
            params = {"W_hidden": W_hidden, "W_out": W_out}
            _, loss = evaluate_binary_mse(mlp_forward_proba, params, X, y)
            curve_epochs.append(ep)
            curve_losses.append(float(loss))

    return {"W_hidden": W_hidden, "W_out": W_out}, curve_epochs, curve_losses


def main():
    OUT_DIR = "results/classic_extended"
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    TAB_DIR = os.path.join(OUT_DIR, "tables")
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(TAB_DIR)

    # -----------------------------
    # Experiment grid (bigger)
    # -----------------------------
    sigma_list = [0.00, 0.10, 0.20, 0.30]
    n_list     = [50, 100, 250, 500]
    data_seeds = [0, 1]             # dataset variability
    model_seeds = [0, 1, 2]         # init variability
    split_seed = 42

    # Training configs
    linear_train = {"epochs": 2000, "lr": 0.1}
    mlp_train = {"epochs": 3000, "lr": 0.3}
    hidden_list = [1, 2, 4, 8]

    # -----------------------------
    # 1) Scatter plots (several)
    # -----------------------------
    scatters = [
        (0.10, 100, 0),
        (0.20, 100, 0),
        (0.10, 250, 1),
    ]

    for sigma, n_per_cluster, seed in scatters:
        X_demo, y_demo = make_noisy_xor(n_per_cluster=n_per_cluster, noise=sigma, seed=seed)
        plot_dataset_scatter(
            X_demo, y_demo,
            title=f"Dataset B Scatter | sigma={sigma:.2f}, n_per_cluster={n_per_cluster}, seed={seed}",
            save_path=os.path.join(FIG_DIR, f"scatter_B_sigma{sigma:.2f}_n{n_per_cluster}_seed{seed}.png")
        )

    # -----------------------------
    # 2) Full raw sweep (Linear + MLP with multiple hidden sizes)
    #    Decision boundaries only for a small showcase set (NOT all)
    # -----------------------------
    all_rows = []

    boundary_targets = [
        # Required-like boundaries:
        (0.10, 100, "linear", -1),
        (0.10, 100, "mlp", 4),

        # Extra showcase boundaries:
        (0.20, 100, "mlp", 4),
        (0.10, 250, "mlp", 4),
        (0.30, 50, "linear", -1),
        (0.40, 50, "mlp", 8),
    ]

    for sigma in sigma_list:
        for n_per_cluster in n_list:
            for ds in data_seeds:
                # --- Linear ---
                for sm in model_seeds:
                    train_cfg = dict(linear_train, seed_model=sm)
                    row, (X, y), _, params = run_one_experiment_B(
                        n_per_cluster, sigma, ds, split_seed,
                        "linear", {}, train_cfg
                    )
                    all_rows.append(row)

                    # Save boundaries only for a few representative conditions
                    if (sigma, n_per_cluster, "linear", -1) in boundary_targets and ds == 0 and sm == 0:
                        plot_decision_boundary(
                            linear_forward_proba, params, X, y,
                            title=f"Decision Boundary | Linear | sigma={sigma:.2f}, n={n_per_cluster}",
                            save_path=os.path.join(FIG_DIR, f"boundary_linear_sigma{sigma:.2f}_n{n_per_cluster}.png")
                        )

                # --- MLP (multiple hidden sizes) ---
                for nh in hidden_list:
                    for sm in model_seeds:
                        train_cfg = dict(mlp_train, seed_model=sm)
                        row, (X, y), _, params = run_one_experiment_B(
                            n_per_cluster, sigma, ds, split_seed,
                            "mlp", {"n_hidden": nh}, train_cfg
                        )
                        all_rows.append(row)

                        if (sigma, n_per_cluster, "mlp", nh) in boundary_targets and ds == 0 and sm == 0:
                            plot_decision_boundary(
                                mlp_forward_proba, params, X, y,
                                title=f"Decision Boundary | MLP(h={nh}) | sigma={sigma:.2f}, n={n_per_cluster}",
                                save_path=os.path.join(FIG_DIR, f"boundary_mlp_h{nh}_sigma{sigma:.2f}_n{n_per_cluster}.png")
                            )

            print(f"[done] sigma={sigma:.2f}, n_per_cluster={n_per_cluster}")

    # Save raw metrics
    raw_path = os.path.join(TAB_DIR, "metrics.csv")
    save_csv_rows(raw_path, list(all_rows[0].keys()), all_rows)

    # -----------------------------
    # 3) Learning curve: MLP(h=4) on sigma=0.10, n=100 (one fixed run)
    # -----------------------------
    X0, y0 = make_noisy_xor(n_per_cluster=100, noise=0.10, seed=0)
    Xtr, ytr, _, _ = train_test_split(X0, y0, train_ratio=0.8, seed=split_seed)

    _, curve_ep, curve_loss = train_mlp_with_curve_mse(
        Xtr, ytr, n_hidden=4, epochs=3000, lr=0.3, seed=0, record_every=10
    )
    plot_curve(
        x_vals=curve_ep,
        y_vals_dict={"MLP(h=4) train MSE": curve_loss},
        title="Learning Curve | MLP(h=4) | Dataset B (sigma=0.10, n_per_cluster=100)",
        xlabel="epoch",
        ylabel="MSE loss",
        save_path=os.path.join(FIG_DIR, "learning_curve_mlp_h4_sigma0.10_n100.png")
    )

    # -----------------------------
    # 4) Accuracy vs noise (fixed n=100): Linear vs MLP(h=4)
    # -----------------------------
    noise_rows = []
    y_lin_means = []
    y_mlp_means = []

    for sigma in sigma_list:
        lin_vals = [
            r["test_acc"] for r in all_rows
            if r["model"] == "linear"
            and r["n_per_cluster"] == 100
            and abs(r["sigma"] - sigma) < 1e-12
        ]
        mlp_vals = [
            r["test_acc"] for r in all_rows
            if r["model"] == "mlp"
            and r["n_hidden"] == 4
            and r["n_per_cluster"] == 100
            and abs(r["sigma"] - sigma) < 1e-12
        ]

        lin_m, lin_s = mean_std(lin_vals)
        mlp_m, mlp_s = mean_std(mlp_vals)

        noise_rows.append({
            "sigma": float(sigma),
            "linear_test_acc_mean": lin_m,
            "linear_test_acc_std": lin_s,
            "mlp_h4_test_acc_mean": mlp_m,
            "mlp_h4_test_acc_std": mlp_s,
        })

        y_lin_means.append(lin_m)
        y_mlp_means.append(mlp_m)

    save_csv_rows(
        os.path.join(TAB_DIR, "table_noise_robustness.csv"),
        list(noise_rows[0].keys()),
        noise_rows
    )

    plot_curve(
        x_vals=sigma_list,
        y_vals_dict={
            "Linear (mean)": y_lin_means,
            "MLP(h=4) (mean)": y_mlp_means
        },
        title="Test Accuracy vs Noise | Dataset B (n_per_cluster=100)",
        xlabel="sigma",
        ylabel="test accuracy (mean)",
        save_path=os.path.join(FIG_DIR, "acc_vs_noise_n100_linear_vs_mlp4.png")
    )

    # -----------------------------
    # 5) Accuracy vs dataset size (fixed sigma=0.10): Linear vs MLP(h=4)
    # -----------------------------
    sig_fixed = 0.10
    size_rows = []
    y_lin = []
    y_mlp = []

    for n_per_cluster in n_list:
        lin_vals = [
            r["test_acc"] for r in all_rows
            if r["model"] == "linear"
            and r["n_per_cluster"] == n_per_cluster
            and abs(r["sigma"] - sig_fixed) < 1e-12
        ]
        mlp_vals = [
            r["test_acc"] for r in all_rows
            if r["model"] == "mlp"
            and r["n_hidden"] == 4
            and r["n_per_cluster"] == n_per_cluster
            and abs(r["sigma"] - sig_fixed) < 1e-12
        ]

        lin_m, lin_s = mean_std(lin_vals)
        mlp_m, mlp_s = mean_std(mlp_vals)

        size_rows.append({
            "n_per_cluster": int(n_per_cluster),
            "linear_test_acc_mean": lin_m,
            "linear_test_acc_std": lin_s,
            "mlp_h4_test_acc_mean": mlp_m,
            "mlp_h4_test_acc_std": mlp_s,
        })

        y_lin.append(lin_m)
        y_mlp.append(mlp_m)

    save_csv_rows(
        os.path.join(TAB_DIR, "table_size_study.csv"),
        list(size_rows[0].keys()),
        size_rows
    )

    plot_curve(
        x_vals=n_list,
        y_vals_dict={
            "Linear (mean)": y_lin,
            "MLP(h=4) (mean)": y_mlp
        },
        title="Test Accuracy vs Dataset Size | Dataset B (sigma=0.10)",
        xlabel="n_per_cluster",
        ylabel="test accuracy (mean)",
        save_path=os.path.join(FIG_DIR, "acc_vs_size_sigma0.10_linear_vs_mlp4.png")
    )

    # -----------------------------
    # 6) Hidden units study (sigma=0.10, n=100): table + plot
    # -----------------------------
    hidden_rows = []
    hidden_means = []

    for nh in hidden_list:
        vals = [
            r["test_acc"] for r in all_rows
            if r["model"] == "mlp"
            and r["n_hidden"] == nh
            and r["n_per_cluster"] == 100
            and abs(r["sigma"] - 0.10) < 1e-12
        ]
        m, s = mean_std(vals)
        hidden_rows.append({
            "n_hidden": int(nh),
            "test_acc_mean": m,
            "test_acc_std": s,
        })
        hidden_means.append(m)

    save_csv_rows(
        os.path.join(TAB_DIR, "table_hidden_units_study.csv"),
        list(hidden_rows[0].keys()),
        hidden_rows
    )

    plot_curve(
        x_vals=hidden_list,
        y_vals_dict={"MLP test acc (mean)": hidden_means},
        title="Hidden Units Study | MLP | Dataset B (sigma=0.10, n=100)",
        xlabel="n_hidden",
        ylabel="test accuracy (mean)",
        save_path=os.path.join(FIG_DIR, "hidden_units_study_mlp_sigma0.10_n100.png")
    )

    # -----------------------------
    # 7) Model summary table (sigma=0.10, n=100): Linear + MLP(h=4)
    # -----------------------------
    summary_rows = []

    # Linear
    lin_subset = [
        r for r in all_rows
        if r["model"] == "linear"
        and r["n_per_cluster"] == 100
        and abs(r["sigma"] - 0.10) < 1e-12
    ]
    lin_acc = [r["test_acc"] for r in lin_subset]
    lin_time = [r["train_time_sec"] for r in lin_subset]
    lin_m, lin_s = mean_std(lin_acc)
    lin_tm, lin_ts = mean_std(lin_time)

    summary_rows.append({
        "model": "Linear",
        "n_params": 3,
        "test_acc_mean": lin_m,
        "test_acc_std": lin_s,
        "train_time_mean_sec": lin_tm,
        "train_time_std_sec": lin_ts,
    })

    # MLP(h=4)
    mlp_subset = [
        r for r in all_rows
        if r["model"] == "mlp"
        and r["n_hidden"] == 4
        and r["n_per_cluster"] == 100
        and abs(r["sigma"] - 0.10) < 1e-12
    ]
    mlp_acc = [r["test_acc"] for r in mlp_subset]
    mlp_time = [r["train_time_sec"] for r in mlp_subset]
    mlp_m, mlp_s = mean_std(mlp_acc)
    mlp_tm, mlp_ts = mean_std(mlp_time)

    n_params_mlp = int(mlp_subset[0]["n_params"]) if mlp_subset else -1
    summary_rows.append({
        "model": "MLP (h=4)",
        "n_params": n_params_mlp,
        "test_acc_mean": mlp_m,
        "test_acc_std": mlp_s,
        "train_time_mean_sec": mlp_tm,
        "train_time_std_sec": mlp_ts,
    })

    save_csv_rows(
        os.path.join(TAB_DIR, "table_model_summary.csv"),
        list(summary_rows[0].keys()),
        summary_rows
    )

    print("\nSaved outputs:")
    print("  Raw metrics CSV:", raw_path)
    print("  Tables folder   :", TAB_DIR)
    print("  Figures folder  :", FIG_DIR)
    print("Done.")


if __name__ == "__main__":
    main()