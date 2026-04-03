from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.analysis import (
    plot_efficiency,
    plot_robustness,
    run_efficiency_metrics,
    run_robustness_analysis,
    write_task3_summary,
)
from src.data import downsample_nodes, load_dataset, split_for_link_prediction
from src.train import TrainConfig, train_one_model


@dataclass
class RunConfig:
    data_root: str
    out_dir: str
    plot_dir: str
    quick: bool
    seed: int
    datasets: List[str]
    models: List[str]
    tune_epochs: int = 2
    efficiency_epochs: int = 12
    robustness_epochs: int = 8
    eval_every: int = 1
    use_amp: bool = True
    use_compile: bool = False
    show_progress: bool = True
    early_stopping_patience: int = 0


def _normalize_models(models: List[str]) -> List[str]:
    allowed = {"gcn", "graphsage", "gat", "lightgcn"}
    normalized = [m.strip().lower() for m in models if m.strip()]
    if not normalized:
        raise ValueError("No models selected. Choose at least one model.")

    unknown = [m for m in normalized if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported model names: {unknown}. Allowed: {sorted(allowed)}")

    # Preserve order but remove duplicates.
    return list(dict.fromkeys(normalized))


def _search_space(model_name: str, quick: bool) -> List[Dict]:
    model_name = model_name.lower()

    if model_name == "lightgcn":
        base = {
            "hidden_dim": [64],
            "num_layers": [2] if quick else [2, 3],
            "dropout": [0.0],
            "lr": [0.01, 0.005],
            "weight_decay": [1e-4],
            "out_dim": [64],
        }
    elif quick:
        base = {
            "hidden_dim": [64],
            "num_layers": [2],
            "dropout": [0.1],
            "lr": [0.01, 0.005],
            "weight_decay": [1e-4],
            "out_dim": [64],
        }
    else:
        base = {
            "hidden_dim": [64, 128],
            "num_layers": [2],
            "dropout": [0.05, 0.2],
            "lr": [0.01, 0.005],
            "weight_decay": [1e-4],
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
        g["gat_heads"] = 1 if model_name == "lightgcn" else 2
    return grid


def tune_model_for_dataset(
    dataset_name: str,
    bundle,
    model_name: str,
    quick: bool,
    seed: int,
    device: torch.device,
    epochs: int,
    eval_every: int,
    use_amp: bool,
    use_compile: bool,
    show_progress: bool,
    early_stopping_patience: int,
) -> Tuple[Dict, Dict, Dict, List[Dict]]:
    best_cfg = None
    best_val_auc = -1.0
    best_val_metrics = None
    best_test_metrics = None
    best_extras = None
    trial_rows: List[Dict] = []

    search_space = _search_space(model_name, quick=quick)
    search_iter = tqdm(
        search_space,
        desc=f"{dataset_name}:{model_name} tune",
        leave=False,
        disable=not show_progress,
    )

    for params in search_iter:
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
            eval_every=eval_every,
            use_amp=use_amp,
            use_compile=use_compile,
            show_progress=show_progress,
            early_stopping_patience=early_stopping_patience,
        )

        val_metrics, test_metrics, extras = train_one_model(
            bundle.train_data,
            bundle.val_data,
            bundle.test_data,
            cfg,
            seed=seed,
            device=device,
        )

        trial_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                **params,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "test_auc": test_metrics["auc"],
                "test_ap": test_metrics["ap"],
                "avg_train_time_per_epoch_s": extras["avg_train_time_per_epoch"],
                "inference_latency_s": extras["inference_latency_s"],
                "num_parameters": extras["num_parameters"],
            }
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_cfg = params
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_extras = extras
            search_iter.set_postfix(best_val_auc=f"{best_val_auc:.4f}")

    assert best_cfg is not None
    assert best_val_metrics is not None
    assert best_test_metrics is not None
    assert best_extras is not None

    return best_cfg, best_val_metrics, {
        **best_test_metrics,
        "avg_train_time_per_epoch_s": best_extras["avg_train_time_per_epoch"],
        "inference_latency_s": best_extras["inference_latency_s"],
        "num_parameters": best_extras["num_parameters"],
    }, trial_rows


def run_all(
    data_root: str,
    out_dir: str,
    plot_dir: str,
    quick: bool,
    seed: int,
    datasets: List[str] | None = None,
    models: List[str] | None = None,
    tune_epochs: int = 2,
    efficiency_epochs: int = 12,
    robustness_epochs: int = 8,
    eval_every: int = 1,
    use_amp: bool = True,
    use_compile: bool = False,
    show_progress: bool = True,
    early_stopping_patience: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if tune_epochs < 1 or efficiency_epochs < 1 or robustness_epochs < 1:
        raise ValueError("Epoch counts must all be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = datasets or ["movielens_1m", "yelp", "amazon_computers"]
    models = _normalize_models(models or ["gcn", "graphsage", "gat", "lightgcn"])

    results_rows = []
    trial_rows: List[Dict] = []
    best_configs: Dict[str, Dict[str, Dict]] = {}
    bundle_sizes: Dict[str, int] = {}
    largest_bundle = None
    largest_name = None

    quick_caps = {
        "movielens_1m": 12000,
        "yelp": 50000,
        "amazon_computers": 14000,
    }

    dataset_iter = tqdm(datasets, desc="Datasets", disable=not show_progress)
    for dataset_name in dataset_iter:
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
        bundle_sizes[dataset_name] = int(data.num_nodes)

        if largest_bundle is None or data.num_nodes > largest_bundle.data.num_nodes:
            largest_bundle = bundle
            largest_name = dataset_name

        model_iter = tqdm(
            models,
            desc=f"{dataset_name} models",
            leave=False,
            disable=not show_progress,
        )
        for model_name in model_iter:
            cfg, val_metrics, test_metrics, model_trials = tune_model_for_dataset(
                dataset_name,
                bundle,
                model_name,
                quick=quick,
                seed=seed,
                device=device,
                epochs=tune_epochs,
                eval_every=eval_every,
                use_amp=use_amp,
                use_compile=use_compile,
                show_progress=show_progress,
                early_stopping_patience=early_stopping_patience,
            )
            best_configs[dataset_name][model_name] = cfg
            trial_rows.extend(model_trials)

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

    task2_df = pd.DataFrame(results_rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    task2_csv = os.path.join(out_dir, "task2_empirical_comparison.csv")
    task2_df.to_csv(task2_csv, index=False)

    tuning_trials_csv = os.path.join(out_dir, "task2_tuning_trials.csv")
    pd.DataFrame(trial_rows).to_csv(tuning_trials_csv, index=False)

    task2_cfg_json = os.path.join(out_dir, "task2_best_configs.json")
    with open(task2_cfg_json, "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)

    run_cfg = RunConfig(
        data_root=data_root,
        out_dir=out_dir,
        plot_dir=plot_dir,
        quick=quick,
        seed=seed,
        datasets=datasets,
        models=models,
        tune_epochs=tune_epochs,
        efficiency_epochs=efficiency_epochs,
        robustness_epochs=robustness_epochs,
        eval_every=eval_every,
        use_amp=use_amp,
        use_compile=use_compile,
        show_progress=show_progress,
        early_stopping_patience=early_stopping_patience,
    )
    run_cfg_json = os.path.join(out_dir, "run_config.json")
    run_cfg_payload = asdict(run_cfg)
    run_cfg_payload["device"] = str(device)
    run_cfg_payload["dataset_num_nodes"] = bundle_sizes
    run_cfg_payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with open(run_cfg_json, "w", encoding="utf-8") as f:
        json.dump(run_cfg_payload, f, indent=2)

    assert largest_bundle is not None
    assert largest_name is not None

    largest_model_cfgs = best_configs[largest_name]

    eff_df = run_efficiency_metrics(
        largest_bundle,
        best_configs=largest_model_cfgs,
        device=device,
        seed=seed,
        epochs=efficiency_epochs,
        eval_every=eval_every,
        use_amp=use_amp,
        use_compile=use_compile,
        show_progress=show_progress,
        early_stopping_patience=early_stopping_patience,
    )
    eff_csv = os.path.join(out_dir, "task3_efficiency_metrics.csv")
    eff_df.to_csv(eff_csv, index=False)

    rob_df = run_robustness_analysis(
        largest_bundle,
        best_configs=largest_model_cfgs,
        device=device,
        seed=seed,
        quick=quick,
        epochs=robustness_epochs,
        eval_every=eval_every,
        use_amp=use_amp,
        use_compile=use_compile,
        show_progress=show_progress,
        early_stopping_patience=early_stopping_patience,
    )
    rob_csv = os.path.join(out_dir, "task3_robustness_analysis.csv")
    rob_df.to_csv(rob_csv, index=False)

    plot_efficiency(eff_df, plot_dir)
    plot_robustness(rob_df, plot_dir)

    task3_md = os.path.join(out_dir, "task3_summary.md")
    write_task3_summary(eff_df, rob_df, task3_md)

    summary_lines = [
        "# Task 2 + Task 3 Execution Summary",
        "",
        f"Device: {device}",
        f"Largest dataset selected automatically: {largest_name}",
        f"Models: {', '.join(models)}",
        f"Epochs (tune / efficiency / robustness): {tune_epochs} / {efficiency_epochs} / {robustness_epochs}",
        "",
        "## Task 2 Output Files",
        "- task2_empirical_comparison.csv",
        "- task2_tuning_trials.csv",
        "- task2_best_configs.json",
        "",
        "## Task 3 Output Files",
        "- task3_efficiency_metrics.csv",
        "- task3_robustness_analysis.csv",
        "- task3_summary.md",
        "- plots/*.png",
        "",
        "## Reproducibility",
        "- run_config.json",
    ]
    run_summary_md = os.path.join(out_dir, "run_summary.md")
    with open(run_summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "task2_csv": task2_csv,
        "task2_trials_csv": tuning_trials_csv,
        "task2_best_configs": task2_cfg_json,
        "eff_csv": eff_csv,
        "rob_csv": rob_csv,
        "task3_summary_md": task3_md,
        "run_summary_md": run_summary_md,
        "run_config_json": run_cfg_json,
        "largest_dataset": largest_name,
    }