# 💧 Microplastic Detection in Water (Project V6)

This project is a comprehensive study of applying **YOLO (You Only Look Once)** models to detect microplastics in water samples using deep learning.

## 📂 Project Structure

```
Microplastic-Detection-in-Water/
├── data/                   # Dataset (images + labels)
├── env/                    # Virtual environment
├── models/                 # Trained YOLO models
├── predict/                # Prediction scripts & results
├── requirements.txt        # Project dependencies
├── Scripts/                # Core processing scripts
│   ├── train_yolo.py       # Training pipeline
│   ├── predict.py          # Inference script
│   ├── test_final_model.py # Final model evaluation
│   └── ...                 # Data processing tools
├── results/                # Evaluation results & plots
├── runs/                   # YOLOv8 training outputs
└── SavedModels/            # Model checkpoints
```

## 🛠️ Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YourUsername/Microplastic-Detection-in-Water.git
    cd Microplastic-Detection-in-Water
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv env
    source venv/Scripts/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Training YOLO Models

We have trained multiple YOLOv8 models with different configurations:

-   **YOLOv8n** (Nano) - Fast baseline
-   **YOLOv8s** (Small) - Balanced
-   **YOLOv8m** (Medium) - High accuracy
-   **YOLOv8l** (Large) - Best performance

To train a model (e.g., YOLOv8m):

```bash
python Scripts/train_yolo.py --model yolov8m --data data.yaml
```

**Data Augmentation Configurations:**
-   `exp1`: High Resolution (2K images)
-   `exp2`: Data Augmentation (Mosaic, Copy-Paste)
-   `exp3`: Best Configuration (Optimized)

## 📊 Evaluation

Evaluate the best trained model:

```bash
python Scripts/test_final_model.py
```

This will generate:
-   Confusion Matrix
-   Precision-Recall Curves
-   F1 Score Curves
-   Detection Images

## 🎯 Make Predictions

Run inference on new images or video:

```bash
python Scripts/predict.py --image path/to/image.jpg --model SavedModels/best.pt
```

## 📂 Dataset Description

The dataset contains **1,255 images** of microplastics with annotations.

**Image Specifications:**
-   **Width**: 2560px
-   **Height**: 1920px
-   **Channels**: 3 (RGB)

**Classes:**
1.  Microplastic
2.  Wood Fiber
3.  Pulp
4.  Foam
5.  Glass
6.  PET Fiber

## 📈 Performance Highlights

-   **Best Model**: YOLOv8l (Custom Finetuned)
-   **High Resolution**: 2560x1920 resolution
-   **Advanced Augmentation**: Mosaic, Copy-Paste, Chroma Jittering
-   **Effective for**: Small objects (18-44px), Overlapping particles

## 📄 References

-   [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)