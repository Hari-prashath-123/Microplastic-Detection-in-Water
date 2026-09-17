# Baseline YOLOv8n Error Analysis

## Model
YOLOv8n

## Dataset
Microplastic Detection Dataset

## Validation Metrics

- Precision: 78.50%
- Recall: 62.08%
- mAP@50: 71.86%
- mAP@50-95: 34.75%

## Error Categories

### 1. Missed Detections
- 44 / 200 sampled particles
- Main causes:
  - Very small particles
  - Low contrast
  - Translucent particles

### 2. False Positives
- 6 detections
- Main causes:
  - Air bubbles
  - Specular reflections

### 3. Duplicate / Poor Bounding Boxes
- 4 detections
- Main cause:
  - Elongated fibers
  - Overlapping predictions

## Proposed Improvements

1. Increase input resolution
2. Train for more epochs
3. Experiment with augmentation
4. Tune confidence threshold
5. Tune NMS IoU
6. Compare a larger YOLO model
