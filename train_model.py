"""
SixRay v3 - YOLOv11 Object Detection Training Script
=====================================================
Dataset: SixRay v3 (14,131 images, 640x640)
Classes: Gun, Knife, Pliers, Scissors, Wrench
Format:  YOLOv11 (Roboflow export)

This script trains a YOLOv11 model on the full SixRay dataset
for detecting prohibited items in X-ray security scans.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data.yaml"
PROJECT_NAME = "sixray_model"

# Training hyperparameters
MODEL_SIZE = "yolo11n.pt"   # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
EPOCHS = 100                # Number of training epochs
BATCH_SIZE = 8             # Adjust based on your GPU VRAM (8 for low VRAM, 16-32 for high VRAM)
IMG_SIZE = 640              # Image size (matches dataset preprocessing)
PATIENCE = 20               # Early stopping patience (stops if no improvement for N epochs)
WORKERS = 4                 # Number of data loading workers
DEVICE = "0"                # GPU device ('0' for first GPU, 'cpu' for CPU-only training)

# Advanced settings
OPTIMIZER = "auto"          # Optimizer: SGD, Adam, AdamW, auto
LR0 = 0.01                 # Initial learning rate
LRF = 0.01                 # Final learning rate (fraction of lr0)
WEIGHT_DECAY = 0.0005       # Weight decay for regularization
WARMUP_EPOCHS = 3.0         # Warmup epochs
MOSAIC = 1.0                # Mosaic augmentation probability
MIXUP = 0.0                 # Mixup augmentation probability
AMP = True                  # Automatic Mixed Precision training


def verify_dataset():
    """Verify that the dataset structure is correct before training."""
    print("=" * 60)
    print("  SixRay v3 - Dataset Verification")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        sys.exit(1)

    splits = {"train": "train", "valid": "valid", "test": "test"}
    for split_name, split_dir in splits.items():
        img_dir = PROJECT_DIR / split_dir / "images"
        lbl_dir = PROJECT_DIR / split_dir / "labels"

        if not img_dir.exists():
            print(f"  WARNING: {split_name} images directory not found: {img_dir}")
            continue

        img_count = len(list(img_dir.glob("*.*")))
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split_name:>6s}: {img_count:>6,} images | {lbl_count:>6,} labels")

    print("=" * 60)
    print()


def train():
    """Train YOLOv11 on the SixRay v3 dataset."""

    # Change working directory to project dir so relative paths in data.yaml work
    os.chdir(PROJECT_DIR)

    # Verify dataset
    verify_dataset()

    # Load model
    print(f"Loading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    # Start training
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Device     : {DEVICE}")
    print(f"  Patience   : {PATIENCE}")
    print()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=PROJECT_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        mosaic=MOSAIC,
        mixup=MIXUP,
        amp=AMP,
        verbose=True,
        save=True,
        save_period=10,        # Save checkpoint every 10 epochs
        plots=True,            # Generate training plots
        val=True,              # Run validation after each epoch
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best weights : runs/detect/{PROJECT_NAME}/weights/best.pt")
    print(f"  Last weights : runs/detect/{PROJECT_NAME}/weights/last.pt")
    print(f"  Results      : runs/detect/{PROJECT_NAME}/")
    print("=" * 60)

    return results


def validate():
    """Run validation on the best trained model."""
    best_weights = PROJECT_DIR / "runs" / "detect" / PROJECT_NAME / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"ERROR: Best weights not found at {best_weights}")
        print("Please train the model first.")
        return

    print("\nRunning validation with best weights...")
    model = YOLO(str(best_weights))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=f"{PROJECT_NAME}_val",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("=" * 60)


def resume_training():
    """Resume training from the last checkpoint."""
    last_weights = PROJECT_DIR / "runs" / "detect" / PROJECT_NAME / "weights" / "last.pt"

    if not last_weights.exists():
        print(f"ERROR: No checkpoint found at {last_weights}")
        print("No previous training to resume. Use 'train' mode to start fresh.")
        return

    os.chdir(PROJECT_DIR)
    verify_dataset()

    print(f"Resuming training from: {last_weights}")
    model = YOLO(str(last_weights))
    results = model.train(resume=True)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best weights : runs/detect/{PROJECT_NAME}/weights/best.pt")
    print(f"  Last weights : runs/detect/{PROJECT_NAME}/weights/last.pt")
    print("=" * 60)

    return results


def predict_test():
    """Run predictions on the test set."""
    best_weights = PROJECT_DIR / "runs" / "detect" / PROJECT_NAME / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"ERROR: Best weights not found at {best_weights}")
        print("Please train the model first.")
        return

    print("\nRunning predictions on test set...")
    model = YOLO(str(best_weights))
    results = model.predict(
        source=str(PROJECT_DIR / "test" / "images"),
        imgsz=IMG_SIZE,
        conf=0.25,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=f"{PROJECT_NAME}_predictions",
        exist_ok=True,
    )

    print(f"\nPredictions saved to: runs/detect/{PROJECT_NAME}_predictions/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SixRay v3 - YOLOv11 Training Pipeline")
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["train", "resume", "val", "predict", "all"],
        help="Mode to run: train, resume, val, predict, or all (default: train)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "resume":
        resume_training()
    elif args.mode == "val":
        validate()
    elif args.mode == "predict":
        predict_test()
    elif args.mode == "all":
        train()
        validate()
        predict_test()
