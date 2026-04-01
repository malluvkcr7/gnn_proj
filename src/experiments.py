from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch

from src.analysis import (
    plot_efficiency,
    plot_robustness,
    run_efficiency_metrics,
    run_robustness_analysis,
    write_task3_summary,
)
from src.data import downsample_nodes, load_dataset, split_for_link_prediction
from src.train import TrainConfig, train_one_model


def _search_space(model_name: str, quick: bool) -> List[Dict]:
    if quick:
        base = {
            "hidden_dim": [64],
            "num_layers": [2],
            "dropout": [0.3],
            "lr": [0.01, 0.005],
            "weight_decay": [1e-4],
            "out_dim": [64],
        }
    else:
        base = {
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
            "dropout": [0.2, 0.4],
            "lr": [0.01, 0.005],
            "weight_decay": [1e-4, 5e-4],
            "out_dim": [64],
        }

    keys = list(base.keys())
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*(base[k] for k in keys))]

    if model_name == "gat":
        heads = [2] if quick else [2, 4]
        expanded = []
        for g in grid:
            for h in heads:
                g2 = dict(g)
                g2["gat_heads"] = h
                expanded.append(g2)
        return expanded

    for g in grid:
        g["gat_heads"] = 2
    return grid


def tune_model_for_dataset(
    dataset_name: str,
    bundle,
    model_name: str,
    quick: bool,
    seed: int,
    device: torch.device,
) -> Tuple[Dict, Dict, Dict]:
    best_cfg = None
    best_val_auc = -1.0
    best_val_metrics = None
    best_test_metrics = None
    best_extras = None

    epochs = 10 if quick else 50

    for params in _search_space(model_name, quick=quick):
        cfg = TrainConfig(
            model_name=model_name,
            hidden_dim=params["hidden_dim"],
            out_dim=params["out_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            epochs=epochs,
            gat_heads=params["gat_heads"],
        )

        val_metrics, test_metrics, extras = train_one_model(
            bundle.train_data,
            bundle.val_data,
            bundle.test_data,
            cfg,
            seed=seed,
            device=device,
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_cfg = params
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_extras = extras

    assert best_cfg is not None
    assert best_val_metrics is not None
    assert best_test_metrics is not None
    assert best_extras is not None

    return best_cfg, best_val_metrics, {
        **best_test_metrics,
        "avg_train_time_per_epoch_s": best_extras["avg_train_time_per_epoch"],
        "inference_latency_s": best_extras["inference_latency_s"],
        "num_parameters": best_extras["num_parameters"],
    }


def run_all(
    data_root: str,
    out_dir: str,
    plot_dir: str,
    quick: bool,
    seed: int,
    datasets: List[str] | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = datasets or ["movielens_1m", "yelp", "amazon_computers"]
    models = ["gcn", "graphsage", "gat"]

    results_rows = []
    best_configs: Dict[str, Dict[str, Dict]] = {}
    bundle_sizes: Dict[str, int] = {}
    largest_bundle = None
    largest_name = None

    quick_caps = {
        "movielens_1m": 12000,
        "yelp": 50000,
        "amazon_computers": 14000,
    }

    for dataset_name in datasets:
        best_configs[dataset_name] = {}

        data = load_dataset(dataset_name, root=data_root)
        if quick:
            cap = quick_caps.get(dataset_name, data.num_nodes)
            data = downsample_nodes(data, max_nodes=cap, seed=seed)

        train_data, val_data, test_data = split_for_link_prediction(data)

        class _Bundle:
            pass

        bundle = _Bundle()
        bundle.name = dataset_name
        bundle.data = data
        bundle.train_data = train_data
        bundle.val_data = val_data
        bundle.test_data = test_data
        bundle_sizes[dataset_name] = data.num_nodes

        if largest_bundle is None or data.num_nodes > largest_bundle.data.num_nodes:
            largest_bundle = bundle
            largest_name = dataset_name

        for model_name in models:
            cfg, val_metrics, test_metrics = tune_model_for_dataset(
                dataset_name,
                bundle,
                model_name,
                quick=quick,
                seed=seed,
                device=device,
            )
            best_configs[dataset_name][model_name] = cfg

            results_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "val_auc": val_metrics["auc"],
                    "val_ap": val_metrics["ap"],
                    "test_auc": test_metrics["auc"],
                    "test_ap": test_metrics["ap"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1": test_metrics["f1"],
                    "avg_train_time_per_epoch_s": test_metrics["avg_train_time_per_epoch_s"],
                    "inference_latency_s": test_metrics["inference_latency_s"],
                    "num_parameters": test_metrics["num_parameters"],
                }
            )

    task2_df = pd.DataFrame(results_rows)
    task2_csv = os.path.join(out_dir, "task2_empirical_comparison.csv")
    task2_df.to_csv(task2_csv, index=False)

    with open(os.path.join(out_dir, "task2_best_configs.json"), "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)

    assert largest_bundle is not None
    assert largest_name is not None

    largest_model_cfgs = best_configs[largest_name]

    eff_df = run_efficiency_metrics(
        largest_bundle,
        best_configs=largest_model_cfgs,
        device=device,
        seed=seed,
        quick=quick,
    )
    eff_csv = os.path.join(out_dir, "task3_efficiency_metrics.csv")
    eff_df.to_csv(eff_csv, index=False)

    rob_df = run_robustness_analysis(
        largest_bundle,
        best_configs=largest_model_cfgs,
        device=device,
        seed=seed,
        quick=quick,
    )
    rob_csv = os.path.join(out_dir, "task3_robustness_analysis.csv")
    rob_df.to_csv(rob_csv, index=False)

    plot_efficiency(eff_df, plot_dir)
    plot_robustness(rob_df, plot_dir)

    write_task3_summary(
        eff_df,
        rob_df,
        os.path.join(out_dir, "task3_summary.md"),
    )

    summary_lines = [
        "# Task 2 + Task 3 Execution Summary",
        "",
        f"Device: {device}",
        f"Largest dataset selected automatically: {largest_name}",
        "",
        "## Task 2 Output Files",
        "- task2_empirical_comparison.csv",
        "- task2_best_configs.json",
        "",
        "## Task 3 Output Files",
        "- task3_efficiency_metrics.csv",
        "- task3_robustness_analysis.csv",
        "- task3_summary.md",
        "- plots/*.png",
    ]
    with open(os.path.join(out_dir, "run_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "task2_csv": task2_csv,
        "eff_csv": eff_csv,
        "rob_csv": rob_csv,
        "largest_dataset": largest_name,
    }
