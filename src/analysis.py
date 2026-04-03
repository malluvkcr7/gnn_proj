from __future__ import annotations

import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.train import TrainConfig, train_one_model


def _edge_dropout(data, ratio: float):
    data = copy.deepcopy(data)
    edge_count = data.edge_index.size(1)
    keep = int(edge_count * (1.0 - ratio))
    perm = torch.randperm(edge_count)[:keep]
    data.edge_index = data.edge_index[:, perm]
    return data


def _feature_noise(data, std: float):
    data = copy.deepcopy(data)
    noise = torch.randn_like(data.x) * std
    data.x = data.x + noise
    return data


def _subsample_train_edges(train_data, keep_ratio: float):
    data = copy.deepcopy(train_data)
    total = data.edge_label_index.size(1)
    keep = max(1, int(total * keep_ratio))
    perm = torch.randperm(total)[:keep]
    data.edge_label_index = data.edge_label_index[:, perm]
    data.edge_label = data.edge_label[perm]
    return data


def run_efficiency_metrics(
    largest_bundle,
    best_configs: Dict[str, Dict],
    device: torch.device,
    seed: int,
    epochs: int,
    eval_every: int,
    use_amp: bool,
    use_compile: bool,
    show_progress: bool,
    early_stopping_patience: int,
) -> pd.DataFrame:
    rows = []
    for model_name, cfg in tqdm(
        best_configs.items(),
        desc="Efficiency models",
        leave=False,
        disable=not show_progress,
    ):
        train_cfg = TrainConfig(
            model_name=model_name,
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg.get("out_dim", 64),
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            epochs=epochs,
            gat_heads=cfg.get("gat_heads", 2),
            eval_every=eval_every,
            use_amp=use_amp,
            use_compile=use_compile,
            show_progress=show_progress,
            early_stopping_patience=early_stopping_patience,
        )

        val_metrics, test_metrics, extras = train_one_model(
            largest_bundle.train_data,
            largest_bundle.val_data,
            largest_bundle.test_data,
            train_cfg,
            seed=seed,
            device=device,
        )

        rows.append(
            {
                "model": model_name,
                "dataset": largest_bundle.name,
                "test_auc": test_metrics["auc"],
                "test_ap": test_metrics["ap"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "avg_train_time_per_epoch_s": extras["avg_train_time_per_epoch"],
                "inference_latency_s": extras["inference_latency_s"],
                "num_parameters": extras["num_parameters"],
            }
        )

    return pd.DataFrame(rows)


def run_robustness_analysis(
    largest_bundle,
    best_configs: Dict[str, Dict],
    device: torch.device,
    seed: int,
    quick: bool,
    epochs: int,
    eval_every: int,
    use_amp: bool,
    use_compile: bool,
    show_progress: bool,
    early_stopping_patience: int,
) -> pd.DataFrame:
    rows: List[Dict] = []

    edge_drop_levels = [0.1, 0.3] if quick else [0.1, 0.3, 0.5]
    noise_levels = [0.1, 0.3] if quick else [0.1, 0.3, 0.5]
    train_edge_keep_levels = [1.0, 0.3] if quick else [1.0, 0.3, 0.1]

    for model_name, cfg in tqdm(
        best_configs.items(),
        desc="Robustness models",
        leave=False,
        disable=not show_progress,
    ):
        base_cfg = TrainConfig(
            model_name=model_name,
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg.get("out_dim", 64),
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            epochs=epochs,
            gat_heads=cfg.get("gat_heads", 2),
            eval_every=eval_every,
            use_amp=use_amp,
            use_compile=use_compile,
            show_progress=show_progress,
            early_stopping_patience=early_stopping_patience,
        )

        for r in edge_drop_levels:
            perturbed_train = _edge_dropout(largest_bundle.train_data, r)
            perturbed_val = _edge_dropout(largest_bundle.val_data, r)
            perturbed_test = _edge_dropout(largest_bundle.test_data, r)

            _, test_metrics, _ = train_one_model(
                perturbed_train, perturbed_val, perturbed_test, base_cfg, seed=seed, device=device
            )
            rows.append(
                {
                    "model": model_name,
                    "perturbation": "edge_dropout",
                    "severity": r,
                    "test_auc": test_metrics["auc"],
                    "test_ap": test_metrics["ap"],
                }
            )

        for s in noise_levels:
            perturbed_train = _feature_noise(largest_bundle.train_data, s)
            perturbed_val = _feature_noise(largest_bundle.val_data, s)
            perturbed_test = _feature_noise(largest_bundle.test_data, s)

            _, test_metrics, _ = train_one_model(
                perturbed_train, perturbed_val, perturbed_test, base_cfg, seed=seed, device=device
            )
            rows.append(
                {
                    "model": model_name,
                    "perturbation": "feature_noise",
                    "severity": s,
                    "test_auc": test_metrics["auc"],
                    "test_ap": test_metrics["ap"],
                }
            )

        for k in train_edge_keep_levels:
            reduced_train = _subsample_train_edges(largest_bundle.train_data, k)

            _, test_metrics, _ = train_one_model(
                reduced_train,
                largest_bundle.val_data,
                largest_bundle.test_data,
                base_cfg,
                seed=seed,
                device=device,
            )
            rows.append(
                {
                    "model": model_name,
                    "perturbation": "train_edge_fraction",
                    "severity": k,
                    "test_auc": test_metrics["auc"],
                    "test_ap": test_metrics["ap"],
                }
            )

    return pd.DataFrame(rows)


def plot_efficiency(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        plt.scatter(row["num_parameters"], row["test_auc"], s=120)
        plt.text(row["num_parameters"], row["test_auc"], row["model"])
    plt.xlabel("Trainable Parameters")
    plt.ylabel("Test AUC")
    plt.title("Efficiency Tradeoff: Accuracy vs Model Size")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/efficiency_params_vs_auc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        plt.scatter(row["avg_train_time_per_epoch_s"], row["test_auc"], s=120)
        plt.text(row["avg_train_time_per_epoch_s"], row["test_auc"], row["model"])
    plt.xlabel("Average Train Time per Epoch (s)")
    plt.ylabel("Test AUC")
    plt.title("Efficiency Tradeoff: Accuracy vs Training Time")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/efficiency_time_vs_auc.png", dpi=200)
    plt.close()


def plot_robustness(df: pd.DataFrame, out_dir: str):
    for perturbation in df["perturbation"].unique():
        subset = df[df["perturbation"] == perturbation].sort_values(["model", "severity"])
        plt.figure(figsize=(8, 5))
        for model in subset["model"].unique():
            s = subset[subset["model"] == model]
            plt.plot(s["severity"], s["test_auc"], marker="o", label=model)
        plt.xlabel("Severity")
        plt.ylabel("Test AUC")
        plt.title(f"Robustness under {perturbation}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/robustness_{perturbation}.png", dpi=200)
        plt.close()


def write_task3_summary(eff_df: pd.DataFrame, rob_df: pd.DataFrame, out_file: str):
    best_eff = eff_df.sort_values("test_auc", ascending=False).iloc[0]

    lines = []
    lines.append("# Task 3: Additional Insights on Largest Dataset\n")
    lines.append("## Analysis Type 1: Efficiency Metrics\n")
    lines.append(
        f"Top model by AUC: {best_eff['model']} (AUC={best_eff['test_auc']:.4f}, "
        f"params={int(best_eff['num_parameters'])}, "
        f"train_time/epoch={best_eff['avg_train_time_per_epoch_s']:.4f}s).\n"
    )

    lines.append("## Analysis Type 2: Robustness Analysis\n")
    for perturb in rob_df["perturbation"].unique():
        s = rob_df[rob_df["perturbation"] == perturb]
        grouped = s.groupby("model")["test_auc"].mean().sort_values(ascending=False)
        winner = grouped.index[0]
        lines.append(f"Under {perturb}, best average AUC model: {winner}.\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
