import os
import shutil
from pathlib import Path
import pandas as pd

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Class mapping (0-indexed for YOLO)
CLASS_MAPPING = {
    "Microplastic": 0
}

def convert_csv_to_yolo(split_name="train"):
    """
    Converts annotations from CSV format to YOLO TXT format for a given split.
    """
    csv_file = IMAGES_DIR / split_name / "_annotations.csv"
    images_folder = IMAGES_DIR / split_name
    output_label_dir = LABELS_DIR / split_name

    # Create destination labels directory
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"Warning: Annotation file not found: {csv_file}")
        return

    print(f"\nProcessing '{split_name}' split...")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} annotations across {df['filename'].nunique()} images in {csv_file.name}")

    # Track count
    box_count = 0
    images_with_boxes = set()

    # Group annotations by image filename
    grouped = df.groupby("filename")

    for filename, rows in grouped:
        images_with_boxes.add(filename)
        # Label file name has the same stem as the image file
        txt_filename = Path(filename).stem + ".txt"
        txt_path = output_label_dir / txt_filename

        lines = []
        for _, row in rows.iterrows():
            cls_name = str(row["class"]).strip()
            if cls_name not in CLASS_MAPPING:
                print(f"Unknown class '{cls_name}' in {filename}. Skipping.")
                continue

            class_id = CLASS_MAPPING[cls_name]
            width = float(row["width"])
            height = float(row["height"])
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            # Compute YOLO normalized coordinates
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Clamp coordinates to [0.0, 1.0] to prevent YOLO format warnings
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            bbox_width = max(0.0, min(1.0, bbox_width))
            bbox_height = max(0.0, min(1.0, bbox_height))

            if bbox_width > 0 and bbox_height > 0:
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
                box_count += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # Check for images without annotations (background negative samples)
    all_images = [
        f.name for f in images_folder.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    images_without_boxes = 0
    for img_name in all_images:
        if img_name not in images_with_boxes:
            txt_filename = Path(img_name).stem + ".txt"
            txt_path = output_label_dir / txt_filename
            # Empty file indicates background sample in YOLO
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("")
            images_without_boxes += 1

    print(f"Completed '{split_name}':")
    print(f"  - Images with boxes: {len(images_with_boxes)}")
    print(f"  - Images without boxes (empty label files): {images_without_boxes}")
    print(f"  - Total label files created: {len(images_with_boxes) + images_without_boxes}")
    print(f"  - Total bounding boxes: {box_count}")


def clean_corrupted_structure():
    """Remove erroneous nested directories like Dataset/labels/train/valid if they exist."""
    nested_err = LABELS_DIR / "train" / "valid"
    if nested_err.exists() and nested_err.is_dir():
        print(f"Cleaning up erroneous directory: {nested_err}")
        shutil.rmtree(nested_err)


def main():
    print(f"Dataset root directory: {DATASET_DIR}")
    clean_corrupted_structure()

    # Convert train and valid sets
    for split in ["train", "valid"]:
        convert_csv_to_yolo(split)

    print("\nAll label files successfully generated in YOLO format!")


if __name__ == "__main__":
    main()
