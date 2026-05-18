"""
Dataset preparation: download from Kaggle, convert VOC XML annotations
to YOLO format, and split into train/val sets.

Usage:
    python prepare_dataset.py              # download from Kaggle and prepare
    python prepare_dataset.py --demo       # generate synthetic demo data for testing
"""

import argparse
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

import config


def parse_voc_xml(xml_path):
    """Parse a Pascal VOC XML annotation file and return bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))

    return img_w, img_h, boxes


def voc_to_yolo(img_w, img_h, box):
    """Convert VOC box (xmin, ymin, xmax, ymax) to YOLO format (cx, cy, w, h) normalized."""
    xmin, ymin, xmax, ymax = box
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def convert_annotations():
    """Convert all VOC XML annotations to YOLO .txt format."""
    xml_files = sorted(config.ANNOTATIONS_DIR.glob("*.xml"))
    if not xml_files:
        print(f"  No XML files found in {config.ANNOTATIONS_DIR}")
        return []

    samples = []
    for xml_path in xml_files:
        img_w, img_h, boxes = parse_voc_xml(xml_path)
        stem = xml_path.stem

        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = config.IMAGES_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        yolo_lines = []
        for box in boxes:
            cx, cy, w, h = voc_to_yolo(img_w, img_h, box)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        samples.append((img_path, yolo_lines))

    print(f"  Converted {len(samples)} annotations to YOLO format")
    return samples


def split_dataset(samples):
    """Split samples into train and val sets, copy to processed directories."""
    for d in [config.TRAIN_IMAGES, config.VAL_IMAGES, config.TRAIN_LABELS, config.VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - config.VAL_SPLIT))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    for subset, img_dir, lbl_dir in [
        (train_samples, config.TRAIN_IMAGES, config.TRAIN_LABELS),
        (val_samples, config.VAL_IMAGES, config.VAL_LABELS),
    ]:
        for img_path, yolo_lines in subset:
            shutil.copy2(img_path, img_dir / img_path.name)
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            txt_path.write_text("\n".join(yolo_lines) + "\n")

    print(f"  Train: {len(train_samples)} images, Val: {len(val_samples)} images")
    return len(train_samples), len(val_samples)


def write_dataset_yaml():
    """Write the YOLO dataset.yaml config file."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {config.PROCESSED_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {config.NUM_CLASSES}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    config.DATASET_YAML.write_text(content)
    print(f"  Written {config.DATASET_YAML}")


def generate_demo_data(num_images=30):
    """Generate synthetic car images with license plate boxes for demo/testing."""
    print("\n[Demo Mode] Generating synthetic dataset...")
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    plate_texts = [
        "ABC 1234", "XYZ 9876", "DEF 5678", "GHI 3456",
        "JKL 7890", "MNO 2345", "PQR 6789", "STU 0123",
    ]

    samples = []
    for i in range(num_images):
        h, w = random.choice([(480, 640), (600, 800), (720, 960)])
        img = np.zeros((h, w, 3), dtype=np.uint8)

        bg_color = [random.randint(100, 200) for _ in range(3)]
        img[:] = bg_color

        car_x1 = random.randint(w // 8, w // 4)
        car_y1 = random.randint(h // 6, h // 3)
        car_x2 = random.randint(w * 3 // 4, w * 7 // 8)
        car_y2 = random.randint(h * 2 // 3, h * 5 // 6)
        car_color = [random.randint(20, 80) for _ in range(3)]
        cv2.rectangle(img, (car_x1, car_y1), (car_x2, car_y2), car_color, -1)

        plate_w = random.randint(80, 140)
        plate_h = random.randint(30, 50)
        plate_x = random.randint(car_x1 + 20, max(car_x1 + 21, car_x2 - plate_w - 20))
        plate_y = random.randint(car_y2 - plate_h - 40, max(car_y2 - plate_h - 39, car_y2 - 10))
        plate_x2_coord = min(plate_x + plate_w, w - 1)
        plate_y2_coord = min(plate_y + plate_h, h - 1)

        cv2.rectangle(img, (plate_x, plate_y), (plate_x2_coord, plate_y2_coord), (255, 255, 255), -1)
        cv2.rectangle(img, (plate_x, plate_y), (plate_x2_coord, plate_y2_coord), (0, 0, 0), 2)
        text = random.choice(plate_texts)
        cv2.putText(img, text, (plate_x + 5, plate_y2_coord - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        img_name = f"car_{i:04d}.jpg"
        cv2.imwrite(str(config.IMAGES_DIR / img_name), img)

        xml_content = f"""<annotation>
  <filename>{img_name}</filename>
  <size>
    <width>{w}</width>
    <height>{h}</height>
    <depth>3</depth>
  </size>
  <object>
    <name>licence</name>
    <bndbox>
      <xmin>{plate_x}</xmin>
      <ymin>{plate_y}</ymin>
      <xmax>{plate_x2_coord}</xmax>
      <ymax>{plate_y2_coord}</ymax>
    </bndbox>
  </object>
</annotation>"""
        xml_path = config.ANNOTATIONS_DIR / f"car_{i:04d}.xml"
        xml_path.write_text(xml_content)

        cx, cy, bw, bh = voc_to_yolo(w, h, (plate_x, plate_y, plate_x2_coord, plate_y2_coord))
        samples.append((config.IMAGES_DIR / img_name, [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]))

    print(f"  Generated {num_images} synthetic images with annotations")
    return samples


def download_from_kaggle():
    """Download the car plate detection dataset from Kaggle."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / "car-plate-detection.zip"

    print(f"  Downloading {config.KAGGLE_DATASET} from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", config.KAGGLE_DATASET,
                "-p", str(config.RAW_DIR),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Install with: pip install kaggle")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Kaggle download failed: {e.stderr}")
        return False

    if zip_path.exists():
        print("  Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(config.RAW_DIR)
        zip_path.unlink()

    if config.IMAGES_DIR.exists() and config.ANNOTATIONS_DIR.exists():
        n_imgs = len(list(config.IMAGES_DIR.glob("*.*")))
        n_anns = len(list(config.ANNOTATIONS_DIR.glob("*.xml")))
        print(f"  Downloaded {n_imgs} images and {n_anns} annotations")
        return True

    print("  ERROR: Expected images/ and annotations/ folders not found after extraction")
    return False


def main():
    parser = argparse.ArgumentParser(description="Prepare license plate dataset")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data")
    args = parser.parse_args()

    print("=" * 50)
    print("  Dataset Preparation")
    print("=" * 50)

    if args.demo:
        samples = generate_demo_data()
    else:
        if not config.IMAGES_DIR.exists() or not config.ANNOTATIONS_DIR.exists():
            print("\n[1/4] Downloading dataset from Kaggle...")
            if not download_from_kaggle():
                print("\n  Alternatively, run with --demo flag to use synthetic data:")
                print("    python prepare_dataset.py --demo")
                return
            step_offset = 1
        else:
            print(f"\n  Dataset already exists at {config.RAW_DIR}")
            step_offset = 0

        print(f"\n[{2 + step_offset}/4] Converting VOC XML → YOLO format...")
        samples = convert_annotations()

    if not samples:
        print("  No samples to process.")
        return

    step = 3 if not args.demo else 2
    print(f"\n[{step}/{'4' if not args.demo else '3'}] Splitting into train/val sets...")
    split_dataset(samples)

    step += 1
    print(f"\n[{step}/{'4' if not args.demo else '3'}] Writing dataset.yaml...")
    write_dataset_yaml()

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
