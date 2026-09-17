import argparse
import sys
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXP_DIR = BASE_DIR / "models" / "baseline_yolov8n"
RESULTS_DIR = BASE_DIR / "results"


def monitor_training_metrics(exp_dir=DEFAULT_EXP_DIR):
    """
    Task 8: Monitor training performance and record metrics:
    - Box loss (train & val)
    - Classification loss (train & val)
    - Precision
    - Recall
    - mAP@50
    - mAP@50-95
    """
    exp_dir = Path(exp_dir)
    results_csv = exp_dir / "results.csv"

    if not results_csv.exists():
        print(f"Error: results.csv not found at {results_csv}")
        print("Make sure training has been started and at least 1 epoch has completed.")
        return None

    df = pd.read_csv(results_csv)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    print("=" * 65)
    print("        TASK 8: TRAINING MONITORING & METRICS REPORT")
    print("=" * 65)
    print(f"Experiment Directory: {exp_dir.resolve()}")
    print(f"Total Epochs Recorded: {len(df)}")
    print("-" * 65)

    # Identify metric columns
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]
    train_box_loss = "train/box_loss" if "train/box_loss" in df.columns else None
    train_cls_loss = "train/cls_loss" if "train/cls_loss" in df.columns else None
    val_box_loss = "val/box_loss" if "val/box_loss" in df.columns else None
    val_cls_loss = "val/cls_loss" if "val/cls_loss" in df.columns else None
    precision_col = "metrics/precision(B)" if "metrics/precision(B)" in df.columns else None
    recall_col = "metrics/recall(B)" if "metrics/recall(B)" in df.columns else None
    map50_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else None
    map50_95_col = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else None

    # Get latest/best epoch results
    latest = df.iloc[-1]

    # Best epoch based on mAP50 if available
    best_idx = df[map50_col].idxmax() if map50_col else -1
    best_row = df.loc[best_idx]

    prec_val = latest.get(precision_col, 0.0)
    rec_val = latest.get(recall_col, 0.0)
    map50_val = latest.get(map50_col, 0.0)
    map50_95_val = latest.get(map50_95_col, 0.0)

    best_prec = best_row.get(precision_col, 0.0)
    best_rec = best_row.get(recall_col, 0.0)
    best_map50 = best_row.get(map50_col, 0.0)
    best_map50_95 = best_row.get(map50_95_col, 0.0)

    # Print requested formatted output
    print(f"{'Metric':<25} {'Latest Epoch':<18} {'Best Epoch':<18}")
    print("-" * 65)
    if train_box_loss:
        print(f"{'Train Box Loss':<25} {latest[train_box_loss]:<18.4f} {best_row[train_box_loss]:<18.4f}")
    if train_cls_loss:
        print(f"{'Train Class Loss':<25} {latest[train_cls_loss]:<18.4f} {best_row[train_cls_loss]:<18.4f}")
    if val_box_loss:
        print(f"{'Val Box Loss':<25} {latest[val_box_loss]:<18.4f} {best_row[val_box_loss]:<18.4f}")
    if val_cls_loss:
        print(f"{'Val Class Loss':<25} {latest[val_cls_loss]:<18.4f} {best_row[val_cls_loss]:<18.4f}")
    print("-" * 65)
    print(f"{'Precision':<25} {prec_val * 100:>6.2f}%{'':<11} {best_prec * 100:>6.2f}%")
    print(f"{'Recall':<25} {rec_val * 100:>6.2f}%{'':<11} {best_rec * 100:>6.2f}%")
    print(f"{'mAP@50':<25} {map50_val * 100:>6.2f}%{'':<11} {best_map50 * 100:>6.2f}%")
    print(f"{'mAP@50-95':<25} {map50_95_val * 100:>6.2f}%{'':<11} {best_map50_95 * 100:>6.2f}%")
    print("=" * 65)

    # Save summary report to results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "training_metrics_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MICROPLASTIC DETECTION - TRAINING METRICS SUMMARY\n")
        f.write(f"Experiment: {exp_dir.name}\n")
        f.write(f"Total Epochs: {len(df)}\n")
        f.write(f"Best Epoch: {int(best_row.get(epoch_col, best_idx))}\n\n")
        f.write(f"Precision:       {best_prec * 100:.2f}%\n")
        f.write(f"Recall:          {best_rec * 100:.2f}%\n")
        f.write(f"mAP@50:          {best_map50 * 100:.2f}%\n")
        f.write(f"mAP@50-95:       {best_map50_95 * 100:.2f}%\n")
        if train_box_loss:
            f.write(f"Train Box Loss:  {best_row[train_box_loss]:.4f}\n")
        if train_cls_loss:
            f.write(f"Train Cls Loss:  {best_row[train_cls_loss]:.4f}\n")
        if val_box_loss:
            f.write(f"Val Box Loss:    {best_row[val_box_loss]:.4f}\n")
        if val_cls_loss:
            f.write(f"Val Cls Loss:    {best_row[val_cls_loss]:.4f}\n")

    print(f"\n[Saved] Summary report saved to: {report_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor YOLO training metrics.")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_EXP_DIR),
        help="Experiment run directory (default: models/baseline_yolov8n)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor_training_metrics(args.dir)
