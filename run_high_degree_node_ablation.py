import argparse
import json
import os

import pandas as pd
import torch

from src.analysis import plot_robustness, run_robustness_analysis
from src.data import load_all_bundles


DEFAULT_LEVELS = [0.005, 0.001, 0.002, 0.01, 0.05, 0.10]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only high-degree node-removal robustness ablations and merge into robustness CSV."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache directory")
    parser.add_argument("--out-dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--plot-dir", type=str, default="./plots", help="Plot output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--datasets",
        type=str,
        default="movielens_1m,yelp,amazon_computers",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gcn,graphsage,gat,lightgcn",
        help="Comma-separated models from: gcn, graphsage, gat, lightgcn.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=",".join(str(v) for v in DEFAULT_LEVELS),
        help="Comma-separated node-drop fractions, e.g. 0.005,0.001,0.002,0.01,0.05,0.10",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for each ablation run")
    parser.add_argument("--eval-every", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val AUC does not improve for this many evals (0 disables).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--quick", action="store_true", help="Use downsampled quick-mode datasets")
    return parser.parse_args()


def _parse_csv_arg(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    levels = [float(v) for v in _parse_csv_arg(args.levels)]
    datasets = _parse_csv_arg(args.datasets)
    models = [m.lower() for m in _parse_csv_arg(args.models)]

    best_cfg_path = os.path.join(args.out_dir, "task2_best_configs.json")
    if not os.path.exists(best_cfg_path):
        raise FileNotFoundError(
            f"Missing {best_cfg_path}. Run run_all.py first to create best configs."
        )

    with open(best_cfg_path, "r", encoding="utf-8") as f:
        best_configs_all = json.load(f)

    bundles = load_all_bundles(
        dataset_names=datasets,
        root=args.data_root,
        quick=args.quick,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for dataset_name in datasets:
        cfgs = best_configs_all.get(dataset_name)
        if cfgs is None:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {best_cfg_path}. Re-run run_all.py with this dataset."
            )

        filtered_cfgs = {m: cfgs[m] for m in models if m in cfgs}
        if not filtered_cfgs:
            raise ValueError(
                f"No requested models found for dataset '{dataset_name}' in {best_cfg_path}."
            )

        df = run_robustness_analysis(
            bundle=bundles[dataset_name],
            best_configs=filtered_cfgs,
            device=device,
            seed=args.seed,
            quick=args.quick,
            epochs=args.epochs,
            eval_every=args.eval_every,
            use_amp=not args.no_amp,
            use_compile=args.compile,
            show_progress=not args.no_progress,
            early_stopping_patience=args.early_stopping_patience,
            enabled_perturbations=["high_degree_node_removal"],
            high_degree_node_levels_override=levels,
        )
        rows.append(df)

    new_df = pd.concat(rows, ignore_index=True)

    rob_csv = os.path.join(args.out_dir, "task3_robustness_analysis.csv")
    if os.path.exists(rob_csv):
        old_df = pd.read_csv(rob_csv)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["dataset", "model", "perturbation", "severity"],
            keep="last",
        )
    else:
        merged = new_df

    merged = merged.sort_values(["dataset", "model", "perturbation", "severity"]).reset_index(drop=True)
    merged.to_csv(rob_csv, index=False)

    plot_robustness(merged, args.plot_dir)

    print("High-degree node-only ablation complete")
    print({
        "rob_csv": rob_csv,
        "plot_dir": args.plot_dir,
        "levels": levels,
        "datasets": datasets,
        "models": models,
        "rows_added": len(new_df),
        "rows_total": len(merged),
        "device": str(device),
    })


if __name__ == "__main__":
    main()
