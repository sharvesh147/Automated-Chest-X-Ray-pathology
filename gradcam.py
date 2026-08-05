import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from utils import (
    get_first_test_image,
    load_image,
    MODEL_PATH,
    HEATMAP_DIR,
    write_audit_log,
)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(original_path: Path, heatmap, output_path: Path, alpha=0.4):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(original_path))
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(str(output_path), superimposed)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    img_path = get_first_test_image()
    if img_path is None:
        raise FileNotFoundError("No test image found.")

    target_size = (150, 150)
    if model.input_shape is not None:
        target_size = tuple(model.input_shape[1:3])

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction_prob = float(model.predict(img_array)[0][0])
    prediction_label = "PNEUMONIA" if prediction_prob >= 0.5 else "NORMAL"
    confidence = float(prediction_prob if prediction_prob >= 0.5 else 1.0 - prediction_prob) * 100.0

    last_conv_layer_name = next(
        layer.name
        for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    )

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    output_path = HEATMAP_DIR / "heatmap.png"
    overlay_heatmap(img_path, heatmap, output_path)
    write_audit_log(img_path, prediction_label, confidence, disease=prediction_label, source="gradcam.py")

    print("Grad-CAM generated successfully.")
    print(f"Prediction: {prediction_label}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
