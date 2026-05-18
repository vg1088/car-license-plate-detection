"""
Project configuration for Car License Plate Detection.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
ANNOTATIONS_DIR = RAW_DIR / "annotations"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_IMAGES = PROCESSED_DIR / "images" / "train"
VAL_IMAGES = PROCESSED_DIR / "images" / "val"
TRAIN_LABELS = PROCESSED_DIR / "labels" / "train"
VAL_LABELS = PROCESSED_DIR / "labels" / "val"

DATASET_YAML = DATA_DIR / "dataset.yaml"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

# ── Dataset ────────────────────────────────────────────
KAGGLE_DATASET = "andrewmvd/car-plate-detection"
CLASS_NAMES = ["license_plate"]
NUM_CLASSES = 1
VAL_SPLIT = 0.2

# ── Training ───────────────────────────────────────────
BASE_MODEL = "yolov8n.pt"
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
LEARNING_RATE = 0.01
PATIENCE = 10

# ── Inference ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
