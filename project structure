import csv
from pathlib import Path

import numpy as np
from PIL import Image

from utils import (
    PROCESSED_DIR,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    ensure_dataset_structure,
    create_required_directories,
)


def preprocess_image(source_path: Path, destination_path: Path) -> None:
    image = Image.open(source_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    processed_array = np.asarray(image, dtype=np.float32) / 255.0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination_path.with_suffix(".npy"), processed_array)


def build_manifest(records: list[tuple[str, str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["split", "class", "image_name", "processed_path"])
        writer.writerows(records)


def main():
    create_required_directories()
    train_dir, test_dir, val_dir = ensure_dataset_structure()
    manifest_records = []

    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for pattern in IMAGE_EXTENSIONS:
                for source_path in sorted(class_dir.rglob(pattern)):
                    destination = PROCESSED_DIR / split_name / class_dir.name / source_path.stem
                    preprocess_image(source_path, destination)
                    manifest_records.append(
                        (
                            split_name,
                            class_dir.name,
                            source_path.name,
                            str(destination.with_suffix(".npy")),
                        )
                    )

    build_manifest(manifest_records, PROCESSED_DIR / "processed_manifest.csv")
    print("Preprocessing completed successfully.")
    print(f"Processed images saved under: {PROCESSED_DIR}")
    print(f"Manifest saved at: {PROCESSED_DIR / 'processed_manifest.csv'}")
    print(f"Total preprocessed images: {len(manifest_records)}")


if __name__ == "__main__":
    main()
