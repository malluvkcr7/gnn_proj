import argparse

from src.experiments import run_all


def parse_args():
    parser = argparse.ArgumentParser(description="GNN Project Task 2 and Task 3 Runner")
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
        "--quick",
        action="store_true",
        help="Quick run for smoke test and fast iteration with lighter tuning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    outputs = run_all(
        data_root=args.data_root,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        quick=args.quick,
        seed=args.seed,
        datasets=datasets,
    )
    print("Execution complete")
    print(outputs)


if __name__ == "__main__":
    main()
