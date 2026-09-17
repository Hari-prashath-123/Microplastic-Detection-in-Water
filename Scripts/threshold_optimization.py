import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = BASE_DIR / "models" / "exp2_long_training" / "weights" / "best.pt"
VAL_IMAGES_DIR = BASE_DIR / "Dataset" / "images" / "valid"
VAL_LABELS_DIR = BASE_DIR / "Dataset" / "labels" / "valid"
OUTPUT_DIR = BASE_DIR / "results" / "improved" / "exp4_threshold_optimization"


def calculate_iou(box1, box2):
    """Normalized xywh format"""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0.0, xi_max - xi_min)
    inter_h = max(0.0, yi_max - yi_min)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def read_gt_boxes(txt_path):
    if not txt_path.exists():
        return []
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def evaluate_thresholds(
    weights_path=DEFAULT_WEIGHTS,
    thresholds=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    iou_thresh=0.5,
    imgsz=800,
    device="0",
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}")
        return

    print("=" * 80)
    print("     EXPERIMENT 4: DETECTION THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model Candidate:   {weights_path.resolve()}")
    print(f"Validation Set:    {VAL_IMAGES_DIR.resolve()}")
    print(f"Threshold Range:   {thresholds}")
    print("-" * 80)

    # 1. Load model and run inference at lowest threshold (0.05) once to collect raw candidates
    model = YOLO(str(weights_path))
    min_conf = min(thresholds)
    image_files = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")))

    print(f"Running inference across all {len(image_files)} validation images...")
    results = model.predict(
        source=str(VAL_IMAGES_DIR),
        conf=min_conf,
        iou=0.45,
        imgsz=imgsz,
        device=device,
        save=False,
        save_txt=False,
        verbose=False,
    )

    # Cache ground truth and raw predictions per image
    dataset_records = []
    total_ground_truth = 0

    for r in results:
        img_stem = Path(r.path).stem
        gt_path = VAL_LABELS_DIR / f"{img_stem}.txt"
        gt_boxes = read_gt_boxes(gt_path)
        total_ground_truth += len(gt_boxes)

        # Raw detections: [x_center, y_center, w, h, conf]
        preds = []
        if len(r.boxes) > 0:
            for b in r.boxes:
                xywhn = b.xywhn[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().item())
                preds.append({"box": xywhn, "conf": conf})

        dataset_records.append({"image": img_stem, "gt": gt_boxes, "preds": preds})

    print(f"Total Ground Truth Microplastics: {total_ground_truth}")
    print("\nEvaluating metrics across confidence thresholds...\n")

    summary_rows = []

    for conf in thresholds:
        tp = 0
        fp = 0
        fn = 0
        total_detected = 0

        for rec in dataset_records:
            gt_boxes = rec["gt"]
            # Filter detections by current confidence threshold
            filtered_preds = [p for p in rec["preds"] if p["conf"] >= conf]
            total_detected += len(filtered_preds)

            matched_gt = set()
            for p in filtered_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gt in enumerate(gt_boxes):
                    iou = calculate_iou(p["box"], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh:
                    if best_gt_idx not in matched_gt:
                        matched_gt.add(best_gt_idx)
                        tp += 1
                    else:
                        # Duplicate/overlapping detection
                        fp += 1
                else:
                    fp += 1

            fn += (len(gt_boxes) - len(matched_gt))

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        summary_rows.append({
            "Threshold": conf,
            "Detected": total_detected,
            "TP": tp,
            "FP": fp,
            "Missed (FN)": fn,
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1, 2),
        })

    df = pd.DataFrame(summary_rows)

    # Display Table
    print(f"{'Threshold':<11} {'Detected':<10} {'TP':<8} {'FP':<8} {'Missed':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(f"{row['Threshold']:<11.2f} {int(row['Detected']):<10} {int(row['TP']):<8} {int(row['FP']):<8} {int(row['Missed (FN)']):<10} {row['Precision (%)']:>6.2f}%      {row['Recall (%)']:>6.2f}%      {row['F1-Score (%)']:>6.2f}%")
    print("=" * 85)

    # Optimal F1 threshold
    best_f1_row = df.loc[df["F1-Score (%)"].idxmax()]
    print(f"\n[Optimal F1 Operating Point]: Threshold = {best_f1_row['Threshold']:.2f}")
    print(f"  - Peak F1-Score: {best_f1_row['F1-Score (%)']:.2f}% (Precision: {best_f1_row['Precision (%)']:.2f}%, Recall: {best_f1_row['Recall (%)']:.2f}%)")
    print(f"  - Total Detected: {int(best_f1_row['Detected'])}, False Positives: {int(best_f1_row['FP'])}, Missed: {int(best_f1_row['Missed (FN)'])}")

    # Save CSV and Text Reports
    csv_path = OUTPUT_DIR / "threshold_comparison.csv"
    df.to_csv(csv_path, index=False)

    txt_path = OUTPUT_DIR / "threshold_analysis.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT 4: DETECTION THRESHOLD OPTIMIZATION REPORT\n")
        f.write(f"Model Candidate: {weights_path.resolve()}\n")
        f.write(f"Total Validation Ground Truth Microplastics: {total_ground_truth}\n\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\nOptimal F1 Operating Point: Threshold = {best_f1_row['Threshold']:.2f}\n")
        f.write(f"Peak F1-Score: {best_f1_row['F1-Score (%)']:.2f}%\n")
        f.write(f"Precision:     {best_f1_row['Precision (%)']:.2f}%\n")
        f.write(f"Recall:        {best_f1_row['Recall (%)']:.2f}%\n")

    print(f"\n[Saved] CSV saved to: {csv_path}")
    print(f"[Saved] Report saved to: {txt_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection thresholds.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference resolution")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_thresholds(weights_path=args.weights, imgsz=args.imgsz)
