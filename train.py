"""
Train a YOLOv8 model for license plate detection.

Usage:
    python train.py                  # train with default settings
    python train.py --epochs 20      # override epochs
    python train.py --resume         # resume from last checkpoint
"""

import argparse

from ultralytics import YOLO

import config


def main():
    parser = argparse.ArgumentParser(description="Train license plate detection model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--model", type=str, default=config.BASE_MODEL)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 50)
    print("  License Plate Detection — Training")
    print("=" * 50)

    if not config.DATASET_YAML.exists():
        print(f"\n  dataset.yaml not found at {config.DATASET_YAML}")
        print("  Run prepare_dataset.py first.")
        return

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        last_ckpt = config.RESULTS_DIR / "train" / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"  No checkpoint found at {last_ckpt}. Starting fresh.")
            args.resume = False

    print(f"\n  Base model:  {args.model}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch}")
    print(f"  Image size:  {args.img_size}")
    print(f"  Dataset:     {config.DATASET_YAML}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=str(config.DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        lr0=config.LEARNING_RATE,
        patience=config.PATIENCE,
        project=str(config.RESULTS_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    best_model = config.RESULTS_DIR / "train" / "weights" / "best.pt"
    if best_model.exists():
        save_path = config.MODELS_DIR / "plate_detector_best.pt"
        import shutil
        shutil.copy2(best_model, save_path)
        print(f"\n  Best model saved to {save_path}")

    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
