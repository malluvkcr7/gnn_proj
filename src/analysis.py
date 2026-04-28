from __future__ import annotations

import copy
import re
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.utils import subgraph
from tqdm.auto import tqdm

from src.train import TrainConfig, train_one_model


def _edge_dropout(data, ratio: float):
    data = copy.deepcopy(data)
    edge_count = data.edge_index.size(1)
    keep = max(1, int(edge_count * (1.0 - ratio)))
    perm = torch.randperm(edge_count, device=data.edge_index.device)[:keep]
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
    perm = torch.randperm(total, device=data.edge_label_index.device)[:keep]
    data.edge_label_index = data.edge_label_index[:, perm]
    data.edge_label = data.edge_label[perm]
    return data


def _top_degree_nodes(data, ratio: float) -> torch.Tensor:
    num_nodes = int(data.num_nodes)
    remove_count = min(max(1, int(num_nodes * ratio)), max(1, num_nodes - 2))
    deg = torch.bincount(data.edge_index.view(-1), minlength=num_nodes)
    return torch.topk(deg, k=remove_count, largest=True).indices


def _induce_subgraph_with_labels(data, keep_nodes: torch.Tensor):
    edge_device = data.edge_index.device
    kept = keep_nodes.to(device=edge_device, dtype=torch.long).sort().values
    new_data = copy.deepcopy(data)

    new_edge_index, _ = subgraph(kept, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    mapping = torch.full((int(data.num_nodes),), -1, dtype=torch.long, device=edge_device)
    mapping[kept] = torch.arange(kept.numel(), dtype=torch.long, device=edge_device)

    src_old, dst_old = data.edge_label_index
    src_old = src_old.to(edge_device)
    dst_old = dst_old.to(edge_device)
    valid = (mapping[src_old] >= 0) & (mapping[dst_old] >= 0)
    if int(valid.sum()) == 0:
        return None

    new_data.x = data.x[kept.to(data.x.device)]
    new_data.edge_index = new_edge_index
    new_data.edge_label_index = torch.stack([mapping[src_old[valid]], mapping[dst_old[valid]]], dim=0)
    new_data.edge_label = data.edge_label[valid.to(data.edge_label.device)]
    return new_data


def _remove_high_degree_nodes(data, ratio: float):
    removed = _top_degree_nodes(data, ratio)
    keep_mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=removed.device)
    keep_mask[removed] = False
    keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)
    return _induce_subgraph_with_labels(data, keep_nodes)


def _remove_high_degree_edges(data, ratio: float):
    new_data = copy.deepcopy(data)
    edge_count = new_data.edge_index.size(1)
    remove_count = min(max(1, int(edge_count * ratio)), max(1, edge_count - 1))

    src = new_data.edge_index[0]
    dst = new_data.edge_index[1]
    deg = torch.bincount(new_data.edge_index.view(-1), minlength=int(new_data.num_nodes)).float()
    score = deg[src] + deg[dst]

    sorted_idx = torch.argsort(score, descending=True)
    remove_idx = sorted_idx[:remove_count]
    keep_mask = torch.ones(edge_count, dtype=torch.bool, device=new_data.edge_index.device)
    keep_mask[remove_idx] = False
    new_data.edge_index = new_data.edge_index[:, keep_mask]
    return new_data


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

        _, test_metrics, extras = train_one_model(
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
    bundle,
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
    enabled_perturbations: Sequence[str] | None = None,
    high_degree_node_levels_override: Sequence[float] | None = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    enabled = set(enabled_perturbations) if enabled_perturbations is not None else None

    edge_drop_levels = [0.1, 0.3] if quick else [0.1, 0.3, 0.5]
    noise_levels = [0.1, 0.3] if quick else [0.1, 0.3, 0.5]
    train_edge_keep_levels = [1.0, 0.3] if quick else [1.0, 0.3, 0.1]
    high_degree_node_levels = [0.01, 0.03] if quick else [0.01, 0.03, 0.05]
    if high_degree_node_levels_override is not None:
        high_degree_node_levels = [float(x) for x in high_degree_node_levels_override]
    high_degree_edge_levels = [0.1, 0.3] if quick else [0.1, 0.3, 0.5]

    for model_name, cfg in tqdm(
        best_configs.items(),
        desc=f"Robustness models ({bundle.name})",
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

        if enabled is None or "edge_dropout" in enabled:
            for r in edge_drop_levels:
                perturbed_train = _edge_dropout(bundle.train_data, r)
                perturbed_val = _edge_dropout(bundle.val_data, r)
                perturbed_test = _edge_dropout(bundle.test_data, r)

                _, test_metrics, _ = train_one_model(
                    perturbed_train,
                    perturbed_val,
                    perturbed_test,
                    base_cfg,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "perturbation": "edge_dropout",
                        "severity": r,
                        "test_auc": test_metrics["auc"],
                        "test_ap": test_metrics["ap"],
                    }
                )

        if enabled is None or "feature_noise" in enabled:
            for s in noise_levels:
                perturbed_train = _feature_noise(bundle.train_data, s)
                perturbed_val = _feature_noise(bundle.val_data, s)
                perturbed_test = _feature_noise(bundle.test_data, s)

                _, test_metrics, _ = train_one_model(
                    perturbed_train,
                    perturbed_val,
                    perturbed_test,
                    base_cfg,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "perturbation": "feature_noise",
                        "severity": s,
                        "test_auc": test_metrics["auc"],
                        "test_ap": test_metrics["ap"],
                    }
                )

        if enabled is None or "train_edge_fraction" in enabled:
            for k in train_edge_keep_levels:
                reduced_train = _subsample_train_edges(bundle.train_data, k)

                _, test_metrics, _ = train_one_model(
                    reduced_train,
                    bundle.val_data,
                    bundle.test_data,
                    base_cfg,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "perturbation": "train_edge_fraction",
                        "severity": k,
                        "test_auc": test_metrics["auc"],
                        "test_ap": test_metrics["ap"],
                    }
                )

        if enabled is None or "high_degree_edge_removal" in enabled:
            for r in high_degree_edge_levels:
                perturbed_train = _remove_high_degree_edges(bundle.train_data, r)
                perturbed_val = _remove_high_degree_edges(bundle.val_data, r)
                perturbed_test = _remove_high_degree_edges(bundle.test_data, r)

                _, test_metrics, _ = train_one_model(
                    perturbed_train,
                    perturbed_val,
                    perturbed_test,
                    base_cfg,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "perturbation": "high_degree_edge_removal",
                        "severity": r,
                        "test_auc": test_metrics["auc"],
                        "test_ap": test_metrics["ap"],
                    }
                )

        if enabled is None or "high_degree_node_removal" in enabled:
            for r in high_degree_node_levels:
                perturbed_train = _remove_high_degree_nodes(bundle.train_data, r)
                perturbed_val = _remove_high_degree_nodes(bundle.val_data, r)
                perturbed_test = _remove_high_degree_nodes(bundle.test_data, r)

                if perturbed_train is None or perturbed_val is None or perturbed_test is None:
                    continue

                _, test_metrics, _ = train_one_model(
                    perturbed_train,
                    perturbed_val,
                    perturbed_test,
                    base_cfg,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    {
                        "dataset": bundle.name,
                        "model": model_name,
                        "perturbation": "high_degree_node_removal",
                        "severity": r,
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
    has_dataset = "dataset" in df.columns
    dataset_values = sorted(df["dataset"].unique()) if has_dataset else ["all"]

    for dataset_name in dataset_values:
        dataset_df = df[df["dataset"] == dataset_name] if has_dataset else df
        for perturbation in dataset_df["perturbation"].unique():
            subset = dataset_df[dataset_df["perturbation"] == perturbation].sort_values(
                ["model", "severity"]
            )
            plt.figure(figsize=(8, 5))
            for model in subset["model"].unique():
                s = subset[subset["model"] == model]
                plt.plot(s["severity"], s["test_auc"], marker="o", label=model)
            plt.xlabel("Severity")
            plt.ylabel("Test AUC")
            plt.title(f"Robustness on {dataset_name}: {perturbation}")
            plt.legend()
            plt.tight_layout()

            safe_dataset = re.sub(r"[^a-zA-Z0-9_\-]", "_", dataset_name)
            plt.savefig(f"{out_dir}/robustness_{safe_dataset}_{perturbation}.png", dpi=200)
            plt.close()


def write_task3_summary(eff_df: pd.DataFrame, rob_df: pd.DataFrame, out_file: str):
    best_eff = eff_df.sort_values("test_auc", ascending=False).iloc[0]

    lines = []
    lines.append("# Task 3: Additional Insights\n")
    lines.append("## Analysis Type 1: Efficiency Metrics\n")
    lines.append(
        f"Top model by AUC on efficiency dataset ({best_eff['dataset']}): {best_eff['model']} "
        f"(AUC={best_eff['test_auc']:.4f}, "
        f"params={int(best_eff['num_parameters'])}, "
        f"train_time/epoch={best_eff['avg_train_time_per_epoch_s']:.4f}s).\n"
    )

    lines.append("## Analysis Type 2: Robustness Analysis (All Datasets)\n")
    if "dataset" in rob_df.columns:
        datasets = sorted(rob_df["dataset"].unique())
        lines.append(f"Datasets covered: {', '.join(datasets)}.\n")

        for dataset_name in datasets:
            ds = rob_df[rob_df["dataset"] == dataset_name]
            lines.append(f"### {dataset_name}\n")
            for perturb in ds["perturbation"].unique():
                s = ds[ds["perturbation"] == perturb]
                grouped = s.groupby("model")["test_auc"].mean().sort_values(ascending=False)
                winner = grouped.index[0]
                lines.append(f"- Under {perturb}, best average AUC model: {winner}.\n")
    else:
        for perturb in rob_df["perturbation"].unique():
            s = rob_df[rob_df["perturbation"] == perturb]
            grouped = s.groupby("model")["test_auc"].mean().sort_values(ascending=False)
            winner = grouped.index[0]
            lines.append(f"Under {perturb}, best average AUC model: {winner}.\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
