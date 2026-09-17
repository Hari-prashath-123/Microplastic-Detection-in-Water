import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "final_model.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "final_model" / "predictions"


def load_config():
    """Load configuration from config/final_model.yaml or use tested defaults."""
    cfg = {
        "model": str(BASE_DIR / "models" / "exp2_long_training" / "weights" / "best.pt"),
        "image_size": 800,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
            if yaml_cfg:
                if "model" in yaml_cfg:
                    cfg["model"] = str(BASE_DIR / yaml_cfg["model"]) if not Path(yaml_cfg["model"]).is_absolute() else yaml_cfg["model"]
                if "image_size" in yaml_cfg:
                    cfg["image_size"] = int(yaml_cfg["image_size"])
                if "confidence_threshold" in yaml_cfg:
                    cfg["confidence_threshold"] = float(yaml_cfg["confidence_threshold"])
                if "iou_threshold" in yaml_cfg:
                    cfg["iou_threshold"] = float(yaml_cfg["iou_threshold"])
    return cfg


def count_microplastics(
    image_path,
    model_or_weights=None,
    imgsz=None,
    conf_threshold=None,
    iou_threshold=None,
    save_annotated=True,
    output_dir=DEFAULT_OUTPUT_DIR,
    device=None,
    quiet=False,
):
    """
    TASK 15: Final Particle Counter
    Performs detection and quantification:
    - Particle counting
    - Confidence scoring
    - Area calculation (px²)
    - Detailed output formatting
    """
    cfg = load_config()
    model_path = model_or_weights or cfg["model"]
    imgsz = imgsz or cfg["image_size"]
    conf = conf_threshold if conf_threshold is not None else cfg["confidence_threshold"]
    iou = iou_threshold if iou_threshold is not None else cfg["iou_threshold"]

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    if isinstance(model_path, (str, Path)):
        model = YOLO(str(model_path))
    else:
        model = model_path

    # Run inference
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]

    boxes = results.boxes
    count = len(boxes)

    # Read image for annotation
    orig_img = cv2.imread(str(image_path))
    h, w, _ = orig_img.shape

    particles = []
    if count > 0:
        for idx, b in enumerate(boxes, start=1):
            c = float(b.conf[0].cpu().item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
            particles.append({
                "id": idx,
                "confidence": c,
                "area": area,
                "bbox": [x1, y1, x2, y2],
            })
        avg_conf = float(np.mean([p["confidence"] for p in particles])) * 100
    else:
        avg_conf = 0.0

    # Print requested formatted output
    if not quiet:
        print("\n========================================")
        print("       MICROPLASTIC ANALYSIS")
        print("========================================")
        print(f"\nImage: {image_path.name}\n")
        print(f"Microplastics detected: {count}\n")
        print(f"Average confidence: {avg_conf:.1f}%\n")
        print("----------------------------------------")
        print("Particle Details")
        print("----------------------------------------\n")
        if count > 0:
            for p in particles:
                print(f"#{p['id']:<4} Confidence: {p['confidence']:.2f}   Area: {p['area']:<6} px²")
        else:
            print("No microplastic particles detected.")
        print("\n========================================")

    # Annotate image
    annotated_path = None
    if save_annotated and orig_img is not None:
        vis_img = orig_img.copy()

        # Draw bounding boxes & particle tags
        for p in particles:
            x1, y1, x2, y2 = p["bbox"]
            conf_val = p["confidence"]
            area_val = p["area"]
            idx = p["id"]

            # Cyan bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (230, 180, 0), 2)

            # Label badge: #ID (conf, area px²)
            badge = f"#{idx} ({conf_val:.2f}, {area_val}px2)"
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis_img, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(0, y1)), (230, 180, 0), -1)
            cv2.putText(
                vis_img,
                badge,
                (x1 + 3, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Header banner at top
        banner_h = 45
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, vis_img, 0.2, 0, vis_img)

        banner_text = f"MICROPLASTICS: {count} | AVG CONF: {avg_conf:.1f}% | OPTIMAL CONF: {conf:.2f}"
        cv2.putText(
            vis_img,
            banner_text,
            (15, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        annotated_path = output_dir / f"pred_{image_path.name}"
        cv2.imwrite(str(annotated_path), vis_img)

    return {
        "image": image_path.name,
        "count": count,
        "avg_confidence": avg_conf,
        "particles": particles,
        "annotated_path": str(annotated_path) if annotated_path else None,
    }


def analyze_batch(source_dir, output_dir=DEFAULT_OUTPUT_DIR):
    source_dir = Path(source_dir)
    images = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")))
    if not images:
        print(f"No images found in {source_dir}")
        return []

    print(f"\nProcessing {len(images)} images from {source_dir}...")
    cfg = load_config()
    model = YOLO(cfg["model"])

    all_results = []
    for img in images:
        res = count_microplastics(
            img,
            model_or_weights=model,
            imgsz=cfg["image_size"],
            conf_threshold=cfg["confidence_threshold"],
            iou_threshold=cfg["iou_threshold"],
            output_dir=output_dir,
            quiet=True,
        )
        all_results.append(res)
        print(f"  - {img.name}: {res['count']} particles (Avg Conf: {res['avg_confidence']:.1f}%)")

    total_particles = sum(r["count"] for r in all_results)
    avg_conf_overall = np.mean([r["avg_confidence"] for r in all_results if r["count"] > 0]) if total_particles > 0 else 0.0

    print("\n" + "=" * 50)
    print("           BATCH EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Images Evaluated:       {len(all_results)}")
    print(f"Total Microplastics Counted:  {total_particles}")
    print(f"Average Particle Density:     {total_particles / len(all_results):.1f} per image")
    print(f"Overall Average Confidence:   {avg_conf_overall:.1f}%")
    print(f"Predictions Saved To:         {output_dir.resolve()}")
    print("=" * 50)
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Final Microplastic Particle Counter.")
    parser.add_argument(
        "--source",
        type=str,
        default="Dataset/images/valid/c--103-_jpg.rf.2b6feafb66289f2dd300735b293066f7.jpg",
        help="Path to image or directory of images",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.source)
    if src.is_dir():
        analyze_batch(src, output_dir=args.output)
    else:
        count_microplastics(
            src,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=args.output,
        )
