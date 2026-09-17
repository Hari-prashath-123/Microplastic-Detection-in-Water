import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "data.yaml"
OUTPUT_DIR = BASE_DIR / "models"


def train_baseline_model(
    model_name="yolov8n.pt",
    epochs=25,
    batch_size=None,
    img_size=640,
    device=None,
    experiment_name="baseline_yolov8n",
):
    """
    Task 7: Train lightweight baseline YOLO model on microplastic dataset.
    """
    print("=" * 60)
    print("  TASK 7: MICROPLASTIC DETECTION - BASELINE YOLO TRAINING")
    print("=" * 60)

    # 1. Detect hardware
    if device is None:
        if torch.cuda.is_available():
            device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Hardware] GPU detected: {gpu_name} -> Training on CUDA (device=0)")
        else:
            device = "cpu"
            print("[Hardware] No CUDA GPU detected in PyTorch -> Training on CPU")

    if batch_size is None:
        batch_size = 16 if device != "cpu" else 8

    print(f"[Config] Model Architecture: {model_name} (Lightweight)")
    print(f"[Config] Dataset YAML:       {DATA_YAML}")
    print(f"[Config] Epochs:             {epochs}")
    print(f"[Config] Batch Size:         {batch_size}")
    print(f"[Config] Image Size:         {img_size}")
    print(f"[Config] Device:             {device}")
    print(f"[Config] Output Project:     {OUTPUT_DIR / experiment_name}")
    print("-" * 60)

    # 2. Load lightweight pretrained model
    print(f"Loading pretrained {model_name}...")
    model = YOLO(model_name)

    # 3. Start training
    # Note: workers=0 or 2 is recommended on Windows to prevent shared-memory deadlocks
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(OUTPUT_DIR),
        name=experiment_name,
        workers=2 if device != "cpu" else 0,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    best_weight = OUTPUT_DIR / experiment_name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Best Weights Saved: {best_weight}")
    print("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline YOLO detector for microplastics.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: auto 16 for GPU, 8 for CPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('0', 'cpu', etc.)")
    parser.add_argument("--name", type=str, default="baseline_yolov8n", help="Experiment name folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        experiment_name=args.name,
    )
