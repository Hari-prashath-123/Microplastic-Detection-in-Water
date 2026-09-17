import base64
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "final_model.yaml"
DEFAULT_WEIGHTS = BASE_DIR / "models" / "exp2_long_training" / "weights" / "best.pt"
SAMPLE_DIR = BASE_DIR / "Dataset" / "images" / "valid"

app = FastAPI(
    title="Microplastic Detection in Water AI",
    description="Automated AI-powered microplastic detection, counting, and contamination analysis.",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_and_config():
    cfg = {
        "model_path": str(DEFAULT_WEIGHTS),
        "image_size": 800,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                if "model" in data:
                    p = BASE_DIR / data["model"]
                    if p.exists():
                        cfg["model_path"] = str(p)
                if "image_size" in data:
                    cfg["image_size"] = int(data["image_size"])
                if "confidence_threshold" in data:
                    cfg["confidence_threshold"] = float(data["confidence_threshold"])
                if "iou_threshold" in data:
                    cfg["iou_threshold"] = float(data["iou_threshold"])

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model from: {cfg['model_path']} on device={device}")
    model = YOLO(cfg["model_path"])
    return model, cfg, device


# Load model globally on startup
model, default_cfg, device_target = load_model_and_config()


def determine_contamination_level(count: int, total_area: int):
    if count == 0:
        return {"level": "Pristine", "color": "#10B981", "desc": "No detectable microplastic contamination."}
    elif count <= 3:
        return {"level": "Low", "color": "#3B82F6", "desc": "Low trace microplastic particle density."}
    elif count <= 7:
        return {"level": "Moderate", "color": "#F59E0B", "desc": "Noticeable microplastic presence. Filtration advised."}
    elif count <= 12:
        return {"level": "High", "color": "#EF4444", "desc": "High concentration of polymer fragments/fibers."}
    else:
        return {"level": "Severe", "color": "#DC2626", "desc": "Critical microplastic contamination detected."}


def process_detection(img_bgr: np.ndarray, conf_thresh: float, iou_thresh: float, imgsz: int):
    h, w, _ = img_bgr.shape

    # Inference
    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device_target,
        verbose=False,
    )[0]

    boxes = results.boxes
    count = len(boxes)

    particles = []
    total_area = 0
    annotated = img_bgr.copy()

    for idx, b in enumerate(boxes, start=1):
        conf_score = float(b.conf[0].cpu().item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        area = (x2 - x1) * (y2 - y1)
        total_area += area

        particles.append({
            "id": idx,
            "confidence": round(conf_score * 100, 1),
            "area_px": area,
            "bbox": [x1, y1, x2, y2],
        })

        # Draw glowing Cyan box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (235, 180, 0), 2)

        # Label tag
        tag = f"#{idx} ({conf_score:.2f}, {area}px2)"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(0, y1)), (235, 180, 0), -1)
        cv2.putText(
            annotated,
            tag,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    avg_conf = round(float(np.mean([p["confidence"] for p in particles])), 1) if count > 0 else 0.0

    # Header status banner
    banner_h = 44
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    header_text = f"MICROPLASTICS DETECTED: {count}  |  AVG CONF: {avg_conf}%  |  AREA: {total_area:,} px2"
    cv2.putText(
        annotated,
        header_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 245, 255),
        2,
        cv2.LINE_AA,
    )

    # Encode annotated image to JPEG Base64
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    # Encode original image to Base64
    _, orig_buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    original_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode("utf-8")

    contamination = determine_contamination_level(count, total_area)
    inference_speed_ms = round(results.speed.get("inference", 9.4), 1)

    return {
        "count": count,
        "average_confidence": avg_conf,
        "total_area_px": total_area,
        "contamination": contamination,
        "particles": particles,
        "annotated_image": annotated_b64,
        "original_image": original_b64,
        "inference_time_ms": inference_speed_ms,
        "image_dimensions": {"width": w, "height": h},
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": device_target,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model_weights": Path(default_cfg["model_path"]).name,
        "default_config": default_cfg,
    }


@app.get("/api/samples")
def get_sample_images():
    """Return a list of curated water sample images from the validation set."""
    sample_files = list(SAMPLE_DIR.glob("*.jpg"))[:6]
    samples = []
    for f in sample_files:
        samples.append({
            "id": f.stem,
            "filename": f.name,
            "display_name": f"Water Sample: {f.name.split('_')[0]}",
        })
    return samples


@app.get("/api/sample/{sample_id}")
def detect_sample(
    sample_id: str,
    confidence: float = 0.25,
    iou: float = 0.45,
):
    target_files = list(SAMPLE_DIR.glob(f"{sample_id}*.jpg"))
    if not target_files:
        raise HTTPException(status_code=404, detail="Sample image not found")

    img_path = target_files[0]
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read sample image")

    data = process_detection(img, conf_thresh=confidence, iou_thresh=iou, imgsz=default_cfg["image_size"])
    data["filename"] = img_path.name
    return data


@app.post("/api/detect")
async def detect_uploaded_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    data = process_detection(img, conf_thresh=confidence, iou_thresh=iou, imgsz=default_cfg["image_size"])
    data["filename"] = file.filename
    return data


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
