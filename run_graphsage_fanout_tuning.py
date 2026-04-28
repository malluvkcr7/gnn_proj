from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import load_all_bundles
from src.train import TrainConfig, train_one_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune GraphSAGE with different fanout values and update Task 2 results files."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache directory")
    parser.add_argument("--out-dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--datasets",
        type=str,
        default="movielens_1m,yelp,amazon_computers",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--fanouts",
        type=str,
        default="5,10,20,40,80",
        help="Comma-separated fanout caps per source node. Use 0 for full graph.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per fanout trial")
    parser.add_argument("--eval-every", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Stop if val AUC does not improve for this many evals (0 disables).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--quick", action="store_true", help="Use downsampled quick-mode datasets")
    return parser.parse_args()


def _parse_csv_arg(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _cap_out_degree(data, fanout: int, seed: int):
    if fanout <= 0:
        return copy.deepcopy(data)

    new_data = copy.deepcopy(data)
    edge_index = new_data.edge_index
    edge_device = edge_index.device

    # Build per-source edge buckets on CPU for deterministic random sampling.
    src_cpu = edge_index[0].detach().cpu()
    edge_count = int(edge_index.size(1))
    buckets = [[] for _ in range(int(new_data.num_nodes))]
    for edge_id in range(edge_count):
        buckets[int(src_cpu[edge_id])].append(edge_id)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    keep_ids_cpu: List[torch.Tensor] = []
    for ids in buckets:
        if not ids:
            continue
        idx = torch.tensor(ids, dtype=torch.long)
        if idx.numel() > fanout:
            perm = torch.randperm(idx.numel(), generator=gen)[:fanout]
            idx = idx[perm]
        keep_ids_cpu.append(idx)

    if not keep_ids_cpu:
        return new_data

    keep_ids = torch.cat(keep_ids_cpu, dim=0)
    keep_ids = keep_ids.to(device=edge_device)
    new_data.edge_index = edge_index[:, keep_ids]
    return new_data


def _train_cfg_from_best(best_cfg: Dict, args) -> TrainConfig:
    return TrainConfig(
        model_name="graphsage",
        hidden_dim=int(best_cfg["hidden_dim"]),
        out_dim=int(best_cfg.get("out_dim", 64)),
        num_layers=int(best_cfg["num_layers"]),
        dropout=float(best_cfg["dropout"]),
        lr=float(best_cfg["lr"]),
        weight_decay=float(best_cfg["weight_decay"]),
        epochs=args.epochs,
        gat_heads=2,
        eval_every=args.eval_every,
        use_amp=not args.no_amp,
        use_compile=args.compile,
        show_progress=not args.no_progress,
        early_stopping_patience=args.early_stopping_patience,
    )


def main():
    args = parse_args()

    datasets = _parse_csv_arg(args.datasets)
    fanouts = [int(float(x)) for x in _parse_csv_arg(args.fanouts)]
    fanouts = list(dict.fromkeys(fanouts))

    os.makedirs(args.out_dir, exist_ok=True)

    task2_csv_path = os.path.join(args.out_dir, "task2_empirical_comparison.csv")
    trials_csv_path = os.path.join(args.out_dir, "task2_tuning_trials.csv")
    best_cfg_path = os.path.join(args.out_dir, "task2_best_configs.json")

    if not os.path.exists(task2_csv_path) or not os.path.exists(trials_csv_path) or not os.path.exists(best_cfg_path):
        raise FileNotFoundError(
            "Expected task2 output files are missing. Run run_all.py first to generate them."
        )

    with open(best_cfg_path, "r", encoding="utf-8") as f:
        best_cfg_all = json.load(f)

    task2_df = pd.read_csv(task2_csv_path)
    trials_df = pd.read_csv(trials_csv_path)

    bundles = load_all_bundles(
        dataset_names=datasets,
        root=args.data_root,
        quick=args.quick,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_trial_rows = []
    winners = []

    for ds_idx, dataset_name in enumerate(tqdm(datasets, desc="Fanout tuning datasets", disable=args.no_progress)):
        ds_cfg = best_cfg_all.get(dataset_name, {})
        if "graphsage" not in ds_cfg:
            raise KeyError(
                f"Missing graphsage config for dataset '{dataset_name}' in {best_cfg_path}."
            )

        base_best_cfg = ds_cfg["graphsage"]
        cfg = _train_cfg_from_best(base_best_cfg, args)
        bundle = bundles[dataset_name]

        best_record = None
        for fanout in fanouts:
            seed_base = args.seed + ds_idx * 1000 + fanout * 17
            train_data = _cap_out_degree(bundle.train_data, fanout=fanout, seed=seed_base + 1)
            val_data = _cap_out_degree(bundle.val_data, fanout=fanout, seed=seed_base + 2)
            test_data = _cap_out_degree(bundle.test_data, fanout=fanout, seed=seed_base + 3)

            val_metrics, test_metrics, extras = train_one_model(
                train_data,
                val_data,
                test_data,
                cfg,
                seed=seed_base,
                device=device,
            )

            row = {
                "dataset": dataset_name,
                "model": "graphsage",
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "out_dim": cfg.out_dim,
                "gat_heads": 2,
                "fanout": fanout,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "test_auc": test_metrics["auc"],
                "test_ap": test_metrics["ap"],
                "avg_train_time_per_epoch_s": extras["avg_train_time_per_epoch"],
                "inference_latency_s": extras["inference_latency_s"],
                "num_parameters": extras["num_parameters"],
                "trial_type": "graphsage_fanout",
            }
            new_trial_rows.append(row)

            if best_record is None or row["val_auc"] > best_record["val_auc"]:
                best_record = row

        assert best_record is not None

        # Keep the best GraphSAGE score seen so far (historical trials vs this fanout search).
        mask = (task2_df["dataset"] == dataset_name) & (task2_df["model"] == "graphsage")
        previous_val_auc = float(task2_df.loc[mask, "val_auc"].iloc[0]) if mask.any() else None

        hist = trials_df[(trials_df["dataset"] == dataset_name) & (trials_df["model"] == "graphsage")]
        hist_best = None
        if not hist.empty:
            hrow = hist.sort_values("val_auc", ascending=False).iloc[0]
            hist_best = {
                "val_auc": float(hrow["val_auc"]),
                "val_ap": float(hrow["val_ap"]),
                "test_auc": float(hrow["test_auc"]),
                "test_ap": float(hrow["test_ap"]),
                "avg_train_time_per_epoch_s": float(hrow["avg_train_time_per_epoch_s"]),
                "inference_latency_s": float(hrow["inference_latency_s"]),
                "num_parameters": float(hrow["num_parameters"]),
                "fanout": int(hrow["fanout"]) if "fanout" in hrow and not pd.isna(hrow["fanout"]) else None,
            }

        chosen = best_record
        source = "fanout_tuning"
        if hist_best is not None and hist_best["val_auc"] >= float(best_record["val_auc"]):
            chosen = hist_best
            source = "historical_trials"

        improved = previous_val_auc is None or float(chosen["val_auc"]) > previous_val_auc

        # Update best config JSON fanout only when chosen winner has an explicit fanout.
        updated_graphsage_cfg = dict(base_best_cfg)
        if chosen.get("fanout") is not None:
            updated_graphsage_cfg["fanout"] = int(chosen["fanout"])
        elif "fanout" in updated_graphsage_cfg:
            updated_graphsage_cfg.pop("fanout", None)
        best_cfg_all[dataset_name]["graphsage"] = updated_graphsage_cfg

        if mask.any():
            task2_df.loc[mask, "val_auc"] = chosen["val_auc"]
            task2_df.loc[mask, "val_ap"] = chosen["val_ap"]
            task2_df.loc[mask, "test_auc"] = chosen["test_auc"]
            task2_df.loc[mask, "test_ap"] = chosen["test_ap"]
            task2_df.loc[mask, "avg_train_time_per_epoch_s"] = chosen["avg_train_time_per_epoch_s"]
            task2_df.loc[mask, "inference_latency_s"] = chosen["inference_latency_s"]
            task2_df.loc[mask, "num_parameters"] = chosen["num_parameters"]
        else:
            task2_df = pd.concat(
                [
                    task2_df,
                    pd.DataFrame(
                        [
                            {
                                "dataset": dataset_name,
                                "model": "graphsage",
                                "val_auc": chosen["val_auc"],
                                "val_ap": chosen["val_ap"],
                                "test_auc": chosen["test_auc"],
                                "test_ap": chosen["test_ap"],
                                "test_accuracy": float("nan"),
                                "test_f1": float("nan"),
                                "avg_train_time_per_epoch_s": chosen["avg_train_time_per_epoch_s"],
                                "inference_latency_s": chosen["inference_latency_s"],
                                "num_parameters": chosen["num_parameters"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        winners.append(
            {
                "dataset": dataset_name,
                "best_fanout": int(best_record["fanout"]),
                "val_auc": float(best_record["val_auc"]),
                "test_auc": float(best_record["test_auc"]),
                "test_ap": float(best_record["test_ap"]),
                "improved": bool(improved),
                "previous_val_auc": previous_val_auc,
                "chosen_source": source,
                "chosen_val_auc": float(chosen["val_auc"]),
            }
        )

    # Merge trial rows and save.
    new_trials_df = pd.DataFrame(new_trial_rows)
    trials_df = pd.concat([trials_df, new_trials_df], ignore_index=True, sort=False)
    trials_df.to_csv(trials_csv_path, index=False)

    task2_df = task2_df.sort_values(["dataset", "model"]).reset_index(drop=True)
    task2_df.to_csv(task2_csv_path, index=False)

    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg_all, f, indent=2)

    summary_path = os.path.join(args.out_dir, "graphsage_fanout_tuning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": datasets,
                "fanouts": fanouts,
                "epochs": args.epochs,
                "quick": bool(args.quick),
                "device": str(device),
                "winners": winners,
            },
            f,
            indent=2,
        )

    print("GraphSAGE fanout tuning complete")
    print(
        {
            "task2_csv": task2_csv_path,
            "task2_trials_csv": trials_csv_path,
            "task2_best_configs": best_cfg_path,
            "summary": summary_path,
            "device": str(device),
            "winners": winners,
        }
    )


if __name__ == "__main__":
    main()
