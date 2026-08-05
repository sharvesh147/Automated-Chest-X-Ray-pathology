import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from utils import (
    get_first_test_image,
    load_image,
    MODEL_PATH,
    CLASS_NAMES,
    CSV_DIR,
    write_audit_log,
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return tf.keras.models.load_model(MODEL_PATH)


def save_prediction_record(image_path: Path, label: str, confidence: float) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CSV_DIR / "prediction_results.csv"
    with output_file.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if output_file.stat().st_size == 0:
            writer.writerow(["patient_id", "image_path", "prediction", "confidence", "disease"])
        writer.writerow([
            image_path.stem,
            str(image_path),
            label,
            f"{confidence:.1f}",
            label,
        ])
    return output_file


def main():
    model = load_model()
    img_path = get_first_test_image()
    if img_path is None:
        raise FileNotFoundError("No test image found. Run preprocess.py to prepare the dataset.")

    target_size = (150, 150)
    if model.input_shape is not None:
        target_size = tuple(model.input_shape[1:3])

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction_prob = float(model.predict(img_array)[0][0])
    label = CLASS_NAMES[1] if prediction_prob >= 0.5 else CLASS_NAMES[0]
    confidence = float(prediction_prob if prediction_prob >= 0.5 else 1.0 - prediction_prob) * 100.0

    output_file = save_prediction_record(img_path, label, confidence)
    write_audit_log(img_path, label, confidence, disease=label, source="predict.py")

    print("Prediction completed successfully.")
    print(f"Image path: {img_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Saved prediction record to: {output_file}")


if __name__ == "__main__":
    main()
