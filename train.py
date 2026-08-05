import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras import applications

from utils import (
    create_required_directories,
    get_data_generators,
    MODEL_PATH,
    REPORT_DIR,
    CSV_DIR,
    CLASS_NAMES,
)


def build_model(input_shape=(224, 224, 3)):
    try:
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    except Exception:
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )

    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_training_metrics(metrics: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        keys = ["metric", "value"]
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        for name, value in metrics.items():
            writer.writerow([name, f"{value:.4f}"])


def main():
    create_required_directories()
    train_generator, validation_generator, test_generator = get_data_generators()

    model = build_model(input_shape=(224, 224, 3))
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH.with_name("best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1,
    )

    model.save(MODEL_PATH)
    if MODEL_PATH.with_name("best_model.h5").exists():
        model = tf.keras.models.load_model(MODEL_PATH.with_name("best_model.h5"))
        model.save(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    test_generator.reset()
    predicted_prob = model.predict(test_generator, verbose=1).ravel()
    y_true = test_generator.classes
    y_pred = (predicted_prob >= 0.5).astype(int)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with (REPORT_DIR / "classification_report.txt").open("w", encoding="utf-8") as text_file:
        text_file.write(report_text)

    plot_history(history, REPORT_DIR / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, REPORT_DIR / "confusion_matrix.png")
    plot_roc_curve(y_true, predicted_prob, REPORT_DIR / "roc_curve.png")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "roc_auc": auc(*roc_curve(y_true, predicted_prob)[:2]),
    }
    write_training_metrics(metrics, CSV_DIR / "training_metrics.csv")

    print("Training completed successfully.")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Report files saved in: {REPORT_DIR}")
    print(f"Metric CSV saved in: {CSV_DIR / 'training_metrics.csv'}")


if __name__ == "__main__":
    main()
