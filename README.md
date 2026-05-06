# GNN Project: Task 2 + Task 3 (Complete Pipeline)

This project implements your remaining milestones after Task 1.

## Scope Covered

### Task 2: Empirical Comparison (10 marks)
- 3 datasets: MovieLens-1M, Yelp, Amazon Computers
- 4 GNN models: GCN, GraphSAGE, GAT, LightGCN
# GNN Project — Link-Prediction Benchmark

This repository runs link-prediction experiments and additional analyses across standard recommender / graph datasets.

Implemented features
- Datasets: MovieLens-1M, Yelp, Amazon Computers (via PyG)
- Models: GCN, GraphSAGE, GAT, LightGCN
- Tasks: per-dataset hyperparameter tuning, efficiency on the largest dataset, and robustness analyses on all selected datasets
- Metrics: AUC, Average Precision (AP), accuracy, F1; plus timing and parameter counts

Robustness perturbations
- Random edge dropout
- Feature noise injection
- Reduced train-edge supervision
- High-degree edge removal (hub-edge attack)
- High-degree node removal (hub-node attack)

Setup
1. Create and activate a Python 3 virtual environment.
2. Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

Quick start
- Run the default full pipeline (may be long):

```bash
python3 run_all.py
```

- Recommended A6000-friendly run (fast tuning):

```bash
python3 run_all.py --tune-epochs 2 --efficiency-epochs 12 --robustness-epochs 8
```

- Run a single dataset or select models:

```bash
python3 run_all.py --datasets amazon_computers --models gcn,lightgcn --tune-epochs 1
```

- Disable mixed precision or progress bars:

```bash
python3 run_all.py --no-amp --no-progress
```

Outputs
- results/task2_empirical_comparison.csv
- results/task2_tuning_trials.csv
- results/task2_best_configs.json
- results/task3_efficiency_metrics.csv
- results/task3_robustness_analysis.csv
- results/task3_summary.md
- results/run_summary.md
- results/run_config.json

Plots are written to the directory passed with `--plot-dir` (default `./plots`).

CLI options
- `--datasets`: comma-separated dataset names
- `--models`: comma-separated models (gcn, graphsage, gat, lightgcn)
- `--tune-epochs`, `--efficiency-epochs`, `--robustness-epochs`: epoch counts
- `--eval-every`: validation interval (epochs)
- `--early-stopping-patience`: validation patience (0 disables)
- `--compile`: enable `torch.compile` if available
- `--no-amp`: disable mixed precision on CUDA
- `--no-progress`: disable tqdm progress bars

Notes
- `run_all.py` performs training during hyperparameter tuning and retrains/evaluates best models during the Task 3 analyses; model objects are not persisted to disk by default (metrics and run metadata are saved). Add `--compile` or `--no-amp` based on your GPU and driver setup.



