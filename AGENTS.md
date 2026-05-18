# AGENTS.md

## Cursor Cloud specific instructions

This is a Python-based Deep Learning / Computer Vision project for car license plate detection. See `README.md` for project overview and milestones.

### Environment

- **Python 3.12** with pip; packages are listed in `requirements.txt`.
- PyTorch is installed as **CPU-only** (`torch+cpu`). The `--extra-index-url https://download.pytorch.org/whl/cpu` flag is used during install to get the CPU wheel. No GPU/CUDA is available in the Cloud Agent VM.
- The `ultralytics` package provides the YOLO CLI (`yolo`) and Python API for object detection.
- User-installed scripts (`yolo`, `torchrun`, etc.) are in `~/.local/bin`; ensure `PATH` includes it.

### Running inference

```bash
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model('path/to/image.jpg')"
```

Or use the CLI:

```bash
yolo detect predict model=yolov8n.pt source=path/to/image.jpg
```

### Key notes

- There are no lint, test, or build configurations yet — this is a planning-stage repo with only `README.md` and `requirements.txt`.
- The Kaggle dataset referenced in `README.md` must be downloaded separately (requires Kaggle API credentials).
- For model training, `ultralytics` handles YOLO training pipelines; for custom CNN/TensorFlow approaches, additional dependencies may be needed.
