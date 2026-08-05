import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
ARCHIVE_ROOT = DATA_DIR / "archive" / "chest_xray" / "chest_xray"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORT_DIR = OUTPUT_DIR / "reports"
CSV_DIR = OUTPUT_DIR / "csv"
PROCESSED_DIR = OUTPUT_DIR / "processed"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
AUDIT_LOG_PATH = OUTPUT_DIR / "audit_log.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = MODEL_DIR / "chest_xray_model.h5"

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def dataset_has_images(root: Path) -> bool:
    if not root.exists():
        return False

    for split in ["train", "test", "val"]:
        split_dir = root / split
        if not split_dir.exists() or not split_dir.is_dir():
            return False

        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not subdirs:
            return False

        found_image = False
        for class_dir in subdirs:
            for pattern in IMAGE_EXTENSIONS:
                if any(class_dir.rglob(pattern)):
                    found_image = True
                    break
            if found_image:
                break

        if not found_image:
            return False

    return True


def ensure_dataset_structure():
    if dataset_has_images(DATA_DIR):
        return DATA_DIR / "train", DATA_DIR / "test", DATA_DIR / "val"

    if not ARCHIVE_ROOT.exists():
        raise FileNotFoundError(
            f"Could not find archive dataset at {ARCHIVE_ROOT}. "
            "Make sure the Kaggle chest X-ray dataset is unpacked under dataset/archive/chest_xray/chest_xray."
        )

    for split in ["train", "test", "val"]:
        src = ARCHIVE_ROOT / split
        dst = DATA_DIR / split
        if not src.exists():
            raise FileNotFoundError(f"Missing dataset split: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    return DATA_DIR / "train", DATA_DIR / "test", DATA_DIR / "val"


def get_data_generators():
    train_dir, test_dir, val_dir = ensure_dataset_structure()

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=True,
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def get_first_test_image() -> Optional[Path]:
    train_dir, test_dir, val_dir = ensure_dataset_structure()
    for pattern in IMAGE_EXTENSIONS:
        images = sorted(test_dir.rglob(pattern))
        if images:
            return images[0]
    return None


def load_image(path: Path, target_size=IMG_SIZE):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return arr


def create_required_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def write_audit_log(image_path: Path, prediction: str, confidence: float, disease: str, source: str = "app"):
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = AUDIT_LOG_PATH.exists()
    with AUDIT_LOG_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "source",
                    "image_path",
                    "prediction",
                    "disease",
                    "confidence",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                source,
                str(image_path),
                prediction,
                disease,
                f"{confidence:.1f}",
            ]
        )


def get_dataset_root():
    if dataset_has_images(DATA_DIR):
        return DATA_DIR
    if ARCHIVE_ROOT.exists():
        return ARCHIVE_ROOT
    raise FileNotFoundError("Dataset directory not found.")
