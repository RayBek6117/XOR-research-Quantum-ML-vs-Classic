from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

from src.quantum.data import make_noisy_xor, stratified_split
from src.quantum.preprocess import fit_minmax_scaler, to_angles
from src.quantum.vqc import build_vqc_circuit, mse_loss, prob_class1
from src.quantum.metrics import accuracy


# -----------------------------
# Small local utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def append_row_csv(csv_path: str, row: Dict[str, Any], header_written: bool) -> bool:
    """Append one row to CSV. Returns updated header_written."""
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", index=False, header=(not header_written))
    return True


def decision_boundary_grid(
    X_all: np.ndarray,
    mins: np.ndarray,
    span: np.ndarray,
    circuit,
    params,
    grid_n: int = 160,     # smaller grid to reduce time/memory
    margin: float = 0.25,
):
    x_min = X_all[:, 0].min() - margin
    x_max = X_all[:, 0].max() + margin
    y_min = X_all[:, 1].min() - margin
    y_max = X_all[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_n),
        np.linspace(y_min, y_max, grid_n),
    )

    Zp = np.zeros(xx.shape, dtype=float)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            raw = np.array([xx[i, j], yy[i, j]], dtype=float).reshape(1, 2)
            ang = to_angles(raw, mins, span)[0]
            Zp[i, j] = float(prob_class1(circuit, params, ang))
    return xx, yy, Zp


# -----------------------------
# Config objects
# -----------------------------
@dataclass
class TrainCfg:
    stepsize: float = 0.15
    train_iters: int = 200       # a bit smaller
    log_every: int = 10
    seed_params: int = 0


@dataclass
class DataCfg:
    noise: float
    n_per_cluster: int
    seed_data: int
    split_seed: int


@dataclass
class ModelCfg:
    n_qubits: int = 2
    n_layers: int = 1
    shots_train: Optional[int] = None


def init_params(n_layers: int, n_qubits: int, seed: int) -> pnp.ndarray:
    rng = np.random.default_rng(seed)
    return pnp.array(
        rng.normal(0.0, 1.0, size=(n_layers, n_qubits, 3)),
        requires_grad=True,
    )


# -----------------------------
# One full run helper
# -----------------------------
def run_one_vqc(
    data_cfg: DataCfg,
    model_cfg: ModelCfg,
    train_cfg: TrainCfg,
    record_curve: bool = False,
):
    # Data
    X, y = make_noisy_xor(
        n_per_cluster=data_cfg.n_per_cluster,
        noise=data_cfg.noise,
        seed=data_cfg.seed_data,
    )
    X_train, y_train, X_test, y_test = stratified_split(
        X, y, train_ratio=0.8, seed=data_cfg.split_seed
    )

    # preprocess
    mins, span = fit_minmax_scaler(X_train)
    X_train_a = to_angles(X_train, mins, span)
    X_test_a = to_angles(X_test, mins, span)

    # model
    circuit = build_vqc_circuit(
        n_qubits=model_cfg.n_qubits,
        n_layers=model_cfg.n_layers,
        shots=model_cfg.shots_train,
    )
    params = init_params(model_cfg.n_layers, model_cfg.n_qubits, train_cfg.seed_params)

    opt = qml.AdamOptimizer(stepsize=train_cfg.stepsize)

    it_hist: List[int] = []
    loss_hist: List[float] = []

    t0 = time.time()
    for it in range(1, train_cfg.train_iters + 1):
        params = opt.step(lambda v: mse_loss(circuit, v, X_train_a, y_train), params)
        if record_curve and (it == 1 or it % train_cfg.log_every == 0):
            it_hist.append(it)
            loss_hist.append(float(mse_loss(circuit, params, X_train_a, y_train)))
    t1 = time.time()

    train_time_sec = float(t1 - t0)
    train_acc = float(accuracy(circuit, params, X_train_a, y_train))
    test_acc = float(accuracy(circuit, params, X_test_a, y_test))
    final_train_loss = float(mse_loss(circuit, params, X_train_a, y_train))

    row = {
        "model": "VQC",
        "noise": float(data_cfg.noise),
        "n_per_cluster": int(data_cfg.n_per_cluster),
        "seed_data": int(data_cfg.seed_data),
        "split_seed": int(data_cfg.split_seed),
        "n_layers": int(model_cfg.n_layers),
        "shots_train": str(model_cfg.shots_train),
        "stepsize": float(train_cfg.stepsize),
        "train_iters": int(train_cfg.train_iters),
        "seed_params": int(train_cfg.seed_params),
        "train_time_sec": float(train_time_sec),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "final_train_loss": float(final_train_loss),
        "params_count": int(model_cfg.n_layers * model_cfg.n_qubits * 3),
    }

    artifacts = {
        "X_all": X,
        "y_all": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "mins": mins,
        "span": span,
        "circuit": circuit,
        "params": params,
        "it_hist": it_hist,
        "loss_hist": loss_hist,
        "X_test_a": X_test_a,
    }
    return row, artifacts


# -----------------------------
# Plot helpers
# -----------------------------
def save_learning_curve(fig_path: str, it_hist: List[int], loss_hist: List[float], title: str):
    plt.figure(figsize=(6.6, 4.8))
    plt.plot(it_hist, loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def save_decision_boundary(fig_path: str, title: str, art: Dict[str, Any], grid_n: int = 160):
    xx, yy, Zp = decision_boundary_grid(
        art["X_all"], art["mins"], art["span"], art["circuit"], art["params"],
        grid_n=grid_n, margin=0.25
    )
    plt.figure(figsize=(7.2, 6.0))
    plt.contourf(xx, yy, Zp, levels=25, alpha=0.35)
    plt.contour(xx, yy, Zp, levels=[0.5], linewidths=2)
    plt.scatter(art["X_train"][:, 0], art["X_train"][:, 1], c=art["y_train"],
                edgecolors="k", linewidths=0.8, marker="o", label="Train")
    plt.scatter(art["X_test"][:, 0], art["X_test"][:, 1], c=art["y_test"],
                edgecolors="k", linewidths=0.8, marker="^", label="Test")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_curve(fig_path: str, x_vals, series: Dict[str, List[float]], title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(6.8, 4.9))
    for name, y in series.items():
        plt.plot(x_vals, y, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = "results/quantum_light"
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    # Smaller sweeps (IMPORTANT)
    sigma_list = [0.00, 0.10, 0.30]        # 3 instead of 5
    n_list = [50, 100, 250]               # 3 instead of 5
    seed_data_list = [0, 1]               # 2 instead of 3
    seed_params_list = [0, 1]             # 2 instead of 5
    split_seed = 42

    L_list = [1, 2]                        # keep depth study

    shots_train = None
    shots_eval_list = [None, 128, 1024]

    train_cfg_base = TrainCfg(stepsize=0.15, train_iters=200, log_every=10, seed_params=0)

    # only one learning curve
    curve_target = dict(noise=0.10, n_per_cluster=100, seed_data=0, n_layers=1, seed_params=0)

    # only 2 decision boundaries
    boundary_showcase = [
        dict(noise=0.10, n_per_cluster=100, seed_data=0, n_layers=1, seed_params=0),
        dict(noise=0.10, n_per_cluster=100, seed_data=0, n_layers=2, seed_params=0),
    ]

    # fixed mode for shots study, keep ONLY two inits
    fixed_mode = dict(noise=0.10, n_per_cluster=100, seed_data=0, n_layers=1)
    keep_shots_inits = [0, 1]
    trained_params_for_shots: Dict[int, pnp.ndarray] = {}
    fixed_mode_art = None

    # stream raw metrics
    raw_csv = os.path.join(tab_dir, "vqc_metrics_raw.csv")
    if os.path.exists(raw_csv):
        os.remove(raw_csv)
    header_written = False

    # --- sweep ---
    for n_layers in L_list:
        for sigma in sigma_list:
            for n_per_cluster in n_list:
                for seed_data in seed_data_list:
                    for seed_params in seed_params_list:
                        data_cfg = DataCfg(sigma, n_per_cluster, seed_data, split_seed)
                        model_cfg = ModelCfg(n_qubits=2, n_layers=n_layers, shots_train=shots_train)
                        train_cfg = TrainCfg(
                            stepsize=train_cfg_base.stepsize,
                            train_iters=train_cfg_base.train_iters,
                            log_every=train_cfg_base.log_every,
                            seed_params=seed_params,
                        )

                        record_curve = (
                            abs(sigma - curve_target["noise"]) < 1e-12
                            and n_per_cluster == curve_target["n_per_cluster"]
                            and seed_data == curve_target["seed_data"]
                            and n_layers == curve_target["n_layers"]
                            and seed_params == curve_target["seed_params"]
                        )

                        row, art = run_one_vqc(data_cfg, model_cfg, train_cfg, record_curve=record_curve)
                        header_written = append_row_csv(raw_csv, row, header_written)

                        if record_curve:
                            lc_path = os.path.join(fig_dir, "learning_curve_VQC_L1_sigma0.10_n100.png")
                            save_learning_curve(
                                lc_path, art["it_hist"], art["loss_hist"],
                                title="VQC learning curve (L=1, shots_train=None) | sigma=0.10, n=100"
                            )

                        for sc in boundary_showcase:
                            if (
                                abs(sigma - sc["noise"]) < 1e-12
                                and n_per_cluster == sc["n_per_cluster"]
                                and seed_data == sc["seed_data"]
                                and n_layers == sc["n_layers"]
                                and seed_params == sc["seed_params"]
                            ):
                                db_path = os.path.join(fig_dir, f"decision_boundary_L{n_layers}_sigma{sigma:.2f}_n{n_per_cluster}.png")
                                save_decision_boundary(
                                    db_path,
                                    title=f"VQC decision boundary | L={n_layers}, sigma={sigma:.2f}, n={n_per_cluster}",
                                    art=art,
                                    grid_n=160,
                                )

                        if (
                            abs(sigma - fixed_mode["noise"]) < 1e-12
                            and n_per_cluster == fixed_mode["n_per_cluster"]
                            and seed_data == fixed_mode["seed_data"]
                            and n_layers == fixed_mode["n_layers"]
                            and seed_params in keep_shots_inits
                        ):
                            trained_params_for_shots[seed_params] = art["params"]
                            fixed_mode_art = art

                    print(f"[done] L={n_layers} sigma={sigma:.2f} n={n_per_cluster} seed_data={seed_data}")

    # --- aggregations from CSV (keeps memory OK) ---
    df = pd.read_csv(raw_csv)

    # Noise robustness: fixed n=100, L=1
    noise_rows = []
    noise_means = []
    for sigma in sigma_list:
        sub = df[(df["n_layers"] == 1) & (df["n_per_cluster"] == 100) & (np.abs(df["noise"] - sigma) < 1e-12)]
        vals = sub["test_acc"].tolist()
        m, s = mean_std(vals)
        noise_rows.append({"sigma": float(sigma), "test_acc_mean": m, "test_acc_std": s})
        noise_means.append(m)
    pd.DataFrame(noise_rows).to_csv(os.path.join(tab_dir, "table_noise_robustness.csv"), index=False)
    plot_curve(
        os.path.join(fig_dir, "acc_vs_noise_L1_n100.png"),
        sigma_list,
        {"VQC L=1 (mean)": noise_means},
        "Test Accuracy vs Noise | VQC (L=1) | n=100",
        "sigma",
        "test accuracy (mean)",
    )

    # Size study: fixed sigma=0.10, L=1
    size_rows = []
    size_means = []
    for n in n_list:
        sub = df[(df["n_layers"] == 1) & (np.abs(df["noise"] - 0.10) < 1e-12) & (df["n_per_cluster"] == n)]
        vals = sub["test_acc"].tolist()
        m, s = mean_std(vals)
        size_rows.append({"n_per_cluster": int(n), "test_acc_mean": m, "test_acc_std": s})
        size_means.append(m)
    pd.DataFrame(size_rows).to_csv(os.path.join(tab_dir, "table_size_study.csv"), index=False)
    plot_curve(
        os.path.join(fig_dir, "acc_vs_size_L1_sigma0.10.png"),
        n_list,
        {"VQC L=1 (mean)": size_means},
        "Test Accuracy vs Dataset Size | VQC (L=1) | sigma=0.10",
        "n_per_cluster",
        "test accuracy (mean)",
    )

    # Depth study: L=1 vs L=2 at sigma=0.10 n=100
    depth_rows = []
    depth_means = {}
    for L in [1, 2]:
        sub = df[(df["n_layers"] == L) & (np.abs(df["noise"] - 0.10) < 1e-12) & (df["n_per_cluster"] == 100)]
        vals = sub["test_acc"].tolist()
        times = sub["train_time_sec"].tolist()
        m, s = mean_std(vals)
        tm, ts = mean_std(times)
        depth_rows.append({
            "L": int(L),
            "params_count": int(2 * L * 3),
            "test_acc_mean": m,
            "test_acc_std": s,
            "train_time_mean_sec": tm,
            "train_time_std_sec": ts,
        })
        depth_means[L] = m
    pd.DataFrame(depth_rows).to_csv(os.path.join(tab_dir, "table_depth_study.csv"), index=False)

    plt.figure(figsize=(6.6, 4.8))
    plt.bar(["L=1", "L=2"], [depth_means[1], depth_means[2]])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Depth")
    plt.ylabel("test accuracy (mean)")
    plt.title("Depth study | sigma=0.10, n=100")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "depth_study_sigma0.10_n100.png"), dpi=300)
    plt.close()

    # Shots eval study (only if fixed_mode params exist)
    if fixed_mode_art is not None and len(trained_params_for_shots) > 0:
        X_test_a = fixed_mode_art["X_test_a"]
        y_test = fixed_mode_art["y_test"]

        shots_rows = []
        shots_means = []
        for shots_eval in shots_eval_list:
            circ_eval = build_vqc_circuit(n_qubits=2, n_layers=1, shots=shots_eval)
            accs = []
            for seed_params, params in trained_params_for_shots.items():
                accs.append(float(accuracy(circ_eval, params, X_test_a, y_test)))
            m, s = mean_std(accs)
            shots_rows.append({
                "shots_eval": "None" if shots_eval is None else int(shots_eval),
                "test_acc_mean": m,
                "test_acc_std": s,
            })
            shots_means.append(m)

        pd.DataFrame(shots_rows).to_csv(os.path.join(tab_dir, "table_shots_study.csv"), index=False)

        labels = ["None" if s is None else str(int(s)) for s in shots_eval_list]
        plt.figure(figsize=(6.6, 4.8))
        xs = list(range(len(shots_eval_list)))
        plt.plot(xs, shots_means, marker="o")
        plt.xticks(xs, labels)
        plt.ylim(0.0, 1.0)
        plt.xlabel("shots (evaluation)")
        plt.ylabel("test accuracy (mean)")
        plt.title("Shots study (evaluation) | sigma=0.10, n=100, L=1")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "acc_vs_shots_eval.png"), dpi=300)
        plt.close()

    print("Saved outputs to:", out_dir)
    print("Raw metrics:", raw_csv)
    print("Figures:", fig_dir)
    print("Tables:", tab_dir)


if __name__ == "__main__":
    main()