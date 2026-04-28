import os
import pandas as pd

from src.analysis import plot_efficiency, plot_robustness, write_task3_summary


def main():
    out_dir = "results"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    eff_csv = os.path.join(out_dir, "task3_efficiency_metrics.csv")
    rob_csv = os.path.join(out_dir, "task3_robustness_analysis.csv")
    task3_md = os.path.join(out_dir, "task3_summary.md")

    if not os.path.exists(eff_csv):
        print(f"Missing {eff_csv}; cannot regenerate efficiency plots.")
    else:
        eff_df = pd.read_csv(eff_csv)
        plot_efficiency(eff_df, plot_dir)
        print(f"Wrote efficiency plots to {plot_dir}")

    if not os.path.exists(rob_csv):
        print(f"Missing {rob_csv}; cannot regenerate robustness plots.")
    else:
        rob_df = pd.read_csv(rob_csv)
        plot_robustness(rob_df, plot_dir)
        print(f"Wrote robustness plots to {plot_dir}")

    # Update task3 summary markdown
    try:
        eff_df = pd.read_csv(eff_csv) if os.path.exists(eff_csv) else pd.DataFrame()
        rob_df = pd.read_csv(rob_csv) if os.path.exists(rob_csv) else pd.DataFrame()
        write_task3_summary(eff_df, rob_df, task3_md)
        print(f"Wrote task3 summary to {task3_md}")
    except Exception as e:
        print("Failed to write task3 summary:", e)


if __name__ == '__main__':
    main()
