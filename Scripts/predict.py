import argparse
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = BASE_DIR / "models" / "baseline_yolov8n" / "weights" / "best.pt"
OUTPUT_DIR = BASE_DIR / "results" / "predictions"


def run_inference(
    source_path,
    weights_path=DEFAULT_WEIGHTS,
    conf_threshold=0.25,
    save_txt=True,
    device=None,
):
    """
    Run detection on an image, folder, or video using the trained microplastic YOLO model.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        return

    print("=" * 60)
    print("      MICROPLASTIC DETECTOR - INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Model Weights:      {weights_path.resolve()}")
    print(f"Input Source:       {source_path}")
    print(f"Confidence Thresh:  {conf_threshold}")
    print(f"Output Directory:   {OUTPUT_DIR.resolve()}")
    print("-" * 60)

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path),
        conf=conf_threshold,
        save=True,
        save_txt=save_txt,
        project=str(OUTPUT_DIR.parent),
        name="predictions",
        exist_ok=True,
        device=device,
    )

    total_detections = 0
    print("\nInference Results:")
    for res in results:
        boxes = res.boxes
        num_boxes = len(boxes)
        total_detections += num_boxes
        print(f"  - {Path(res.path).name}: {num_boxes} microplastic particle(s) detected")

    print("-" * 60)
    print(f"Total Detections Across Processed Images: {total_detections}")
    print(f"Visualized annotated outputs saved in:    {OUTPUT_DIR.resolve()}")
    print("=" * 60)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run microplastic detection inference.")
    parser.add_argument(
        "--source",
        type=str,
        default="Dataset/images/valid/c--103-_jpg.rf.2b6feafb66289f2dd300735b293066f7.jpg",
        help="Path to image, folder, or video",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained model weights (default: best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(source_path=args.source, weights_path=args.weights, conf_threshold=args.conf)
