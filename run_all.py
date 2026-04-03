import argparse

from src.experiments import run_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN Project Task 2 + Task 3 runner (GCN/GraphSAGE/GAT/LightGCN)"
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache directory")
    parser.add_argument("--out-dir", type=str, default="./results", help="Output directory")
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
        help="Comma-separated model names from: gcn, graphsage, gat, lightgcn.",
    )
    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=10,
        help="Epochs for hyperparameter tuning runs.",
    )
    parser.add_argument(
        "--efficiency-epochs",
        type=int,
        default=1,
        help="Epochs for efficiency analysis.",
    )
    parser.add_argument(
        "--robustness-epochs",
        type=int,
        default=5,
        help="Epochs for robustness analysis.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Validation interval in epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val AUC does not improve for this many evals (0 disables).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for model forward graph optimization when available.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run for smoke test and fast iteration with lighter search space.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    outputs = run_all(
        data_root=args.data_root,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        quick=args.quick,
        seed=args.seed,
        datasets=datasets,
        models=models,
        tune_epochs=args.tune_epochs,
        efficiency_epochs=args.efficiency_epochs,
        robustness_epochs=args.robustness_epochs,
        eval_every=args.eval_every,
        use_amp=not args.no_amp,
        use_compile=args.compile,
        show_progress=not args.no_progress,
        early_stopping_patience=args.early_stopping_patience,
    )
    print("Execution complete")
    print(outputs)


if __name__ == "__main__":
    main()
