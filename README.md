# GNN Project: Task 2 + Task 3 (Complete Pipeline)

This project implements your remaining milestones after Task 1.

## Scope Covered

### Task 2: Empirical Comparison (10 marks)
- 3 datasets: MovieLens-1M, Yelp, Amazon Computers
- 3 GNN models: GCN, GraphSAGE, GAT
- Link Prediction setup for all datasets
- Hyperparameter tuning for each dataset-model pair
- Benchmark metrics: AUC, AP, Accuracy, F1

### Task 3: Additional Insights (10 marks total, 5+5)
Performed on the largest dataset automatically (by node count):
- Analysis 1: Efficiency Metrics
- Analysis 2: Robustness Analysis

All analyses are run across all three GNN models.

## Setup

1. Create and activate environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

### Quick smoke test

```bash
python run_all.py --quick
```

### Faster smoke test on one dataset

```bash
python run_all.py --quick --datasets amazon_computers
```

### Full run

```bash
python run_all.py
```

You can also choose datasets manually:

```bash
python run_all.py --datasets movielens_1m,yelp,amazon_computers
```

## Outputs

### Results tables
- results/task2_empirical_comparison.csv
- results/task2_best_configs.json
- results/task3_efficiency_metrics.csv
- results/task3_robustness_analysis.csv
- results/task3_summary.md
- results/run_summary.md

### Plots
- plots/efficiency_params_vs_auc.png
- plots/efficiency_time_vs_auc.png
- plots/robustness_edge_dropout.png
- plots/robustness_feature_noise.png
- plots/robustness_train_edge_fraction.png

## Notes for Presentation

Use the outputs directly in your 10-15 minute presentation:
- Task 2 section:
  - Show benchmark table and highlight best model per dataset.
  - Compare how model ranking changes across datasets.
- Task 3 section:
  - Efficiency: show accuracy-cost tradeoff plots.
  - Robustness: show degradation curves under edge dropout, feature noise, and reduced supervision.

## Optional Improvement for Even Stronger Submission

- Run each experiment with 3 different seeds and report mean ± std for AUC/AP.
- Add statistical significance comparison between top two models on each dataset.
