"""
Run license plate detection on individual images.

Usage:
    python detect.py --source image.jpg
    python detect.py --source images_folder/
    python detect.py --source image.jpg --model models/plate_detector_best.pt
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

import config


def detect(source, model_path, save_dir):
    """Run detection and save annotated results."""
    model = YOLO(str(model_path))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = model(
        source=str(source),
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        save=False,
        verbose=False,
    )

    total_detections = 0
    for result in results:
        img_name = Path(result.path).stem
        n_det = len(result.boxes)
        total_detections += n_det

        annotated = result.plot()
        output_path = save_dir / f"{img_name}_detected.jpg"
        cv2.imwrite(str(output_path), annotated)

        print(f"  {Path(result.path).name}: {n_det} plate(s) detected → {output_path.name}")

        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            print(f"    └─ confidence={conf:.2f}  bbox=[{x1}, {y1}, {x2}, {y2}]")

    print(f"\n  Total: {total_detections} plate(s) in {len(results)} image(s)")
    print(f"  Results saved to {save_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect license plates in images")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
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

    save_dir = Path(args.output) if args.output else config.RESULTS_DIR / "detections"

    print("=" * 50)
    print("  License Plate Detection — Inference")
    print("=" * 50)
    print(f"  Model:  {model_path}")
    print(f"  Source: {args.source}")
    print()

    detect(args.source, model_path, save_dir)


if __name__ == "__main__":
    main()
