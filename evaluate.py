"""
Evaluate a trained license plate detection model on the validation set.

Usage:
    python evaluate.py                                  # use best trained model
    python evaluate.py --model models/plate_detector_best.pt
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

import config


def evaluate_model(model_path):
    """Run validation and print metrics."""
    print(f"\n  Model: {model_path}")
    print(f"  Dataset: {config.DATASET_YAML}")

    model = YOLO(str(model_path))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(config.DATASET_YAML),
        project=str(config.RESULTS_DIR),
        name="eval",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")

    return metrics


def visualize_predictions(model_path, num_samples=6):
    """Run detection on sample validation images and save a grid visualization."""
    model = YOLO(str(model_path))
    val_images = sorted(config.VAL_IMAGES.glob("*.*"))

    if not val_images:
        print("  No validation images found.")
        return

    samples = val_images[:num_samples]
    cols = min(3, len(samples))
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        results = model(str(img_path), conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        axes[i].imshow(annotated_rgb)
        n_det = len(results[0].boxes)
        axes[i].set_title(f"{img_path.name} ({n_det} detections)")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path = config.RESULTS_DIR / "predictions_grid.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Prediction grid saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate license plate detection model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--visualize", action="store_true", default=True)
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = config.MODELS_DIR / "plate_detector_best.pt"
        if not model_path.exists():
            model_path = config.RESULTS_DIR / "train" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"  Model not found at {model_path}")
        print("  Train a model first with: python train.py")
        return

    print("=" * 50)
    print("  License Plate Detection — Evaluation")
    print("=" * 50)

    evaluate_model(model_path)

    if args.visualize:
        print("\n  Generating prediction visualizations...")
        visualize_predictions(model_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
