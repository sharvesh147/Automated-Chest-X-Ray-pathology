import csv
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

from report import build_pdf_report
from utils import (
    CLASS_NAMES,
    CSV_DIR,
    HEATMAP_DIR,
    MODEL_PATH,
    REPORT_DIR,
    UPLOAD_DIR,
    create_required_directories,
    load_image,
    write_audit_log,
)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def format_prediction(prediction_prob: float) -> tuple[str, float]:
    if prediction_prob >= 0.5:
        return CLASS_NAMES[1], prediction_prob * 100.0
    return CLASS_NAMES[0], (1.0 - prediction_prob) * 100.0


def save_prediction_csv(patient_id: str, prediction: str, confidence: float) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CSV_DIR / "prediction_report.csv"
    file_exists = output_file.exists()
    with output_file.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or output_file.stat().st_size == 0:
            writer.writerow(["patient_id", "prediction", "confidence", "date", "time"])
        writer.writerow(
            [
                patient_id,
                prediction,
                f"{confidence:.1f}%",
                st.session_state.get("report_date"),
                st.session_state.get("report_time"),
            ]
        )
    return output_file


def main():
    create_required_directories()
    st.set_page_config(page_title="Automated Chest X-Ray Screening", layout="wide")

    st.markdown("# Automated Chest X-Ray Pathology Screening")
    st.markdown(
        "A modern medical dashboard for real-time chest X-ray prediction, Grad-CAM localization, and radiology report export."
    )

    st.sidebar.header("Upload and analyze")
    uploaded_file = st.sidebar.file_uploader("Upload Chest X-Ray image", type=["jpg", "jpeg", "png"])
    st.sidebar.write("\n")
    st.sidebar.write("Upload a chest X-ray image to generate prediction and download the report.")

    model = load_model()
    if model is None:
        st.warning("No trained model found. Run `python train.py` before using the app.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Image Diagnostics")
        if uploaded_file is not None:
            upload_path = UPLOAD_DIR / uploaded_file.name
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            with upload_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())

            original_image = Image.open(upload_path)
            st.image(original_image, caption="Uploaded Chest X-Ray", use_container_width=True)

            if model is None:
                st.error("No trained model found. Run `python train.py` before using the app.")
            else:
                target_size = (150, 150)
                if model.input_shape is not None:
                    target_size = tuple(model.input_shape[1:3])

                model_image = original_image.convert("RGB").resize(target_size)
                temp_array = tf.keras.preprocessing.image.img_to_array(model_image) / 255.0
                temp_array = tf.expand_dims(temp_array, axis=0)
                prediction_prob = float(model.predict(temp_array)[0][0])
                prediction_label, confidence = format_prediction(prediction_prob)
                st.metric("Prediction", prediction_label)
                st.metric("Confidence", f"{confidence:.1f}%")

                st.write("### Radiology Findings")
                st.write(
                    "This system applies deep learning to screen for pneumonia-like pathology and generate explainable Grad-CAM localization overlays."
                )

                if st.button("Save report and export CSV"):
                    st.session_state["report_date"] = st.session_state.get(
                        "report_date",
                        __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
                    )
                    st.session_state["report_time"] = st.session_state.get(
                        "report_time",
                        __import__("datetime").datetime.now().strftime("%H:%M:%S"),
                    )
                    output_file = save_prediction_csv(upload_path.stem, prediction_label, confidence)
                    write_audit_log(upload_path, prediction_label, confidence, disease=prediction_label, source="app.py")
                    st.success(f"Saved prediction and exported CSV to: {output_file}")

                if st.button("Generate PDF report"):
                    report_file = REPORT_DIR / "report.pdf"
                    build_pdf_report(upload_path, prediction_label, confidence, report_file)
                    st.success(f"Generated PDF report: {report_file.name}")

                if st.button("Show Grad-CAM sample heatmap"):
                    if (HEATMAP_DIR / "heatmap.png").exists():
                        st.image(str(HEATMAP_DIR / "heatmap.png"), caption="Sample Grad-CAM heatmap", use_container_width=True)
                    else:
                        st.warning("No Grad-CAM image available. Run gradcam.py first.")
        else:
            st.info("Upload a chest X-ray image from the sidebar to start prediction.")

    with col2:
        st.subheader("Report Download")
        st.write(
            "After prediction, download the latest X-ray analysis report in PDF format."
        )

        report_file = REPORT_DIR / "report.pdf"
        if report_file.exists():
            with report_file.open("rb") as f:
                report_bytes = f.read()
            st.download_button(
                label="Download X-Ray Report PDF",
                data=report_bytes,
                file_name="xray_report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("No report PDF available yet. Run report generation after prediction.")

    st.sidebar.header("Audit & security")
    st.sidebar.write("All predictions are logged to outputs/audit_log.csv for traceability.")


if __name__ == "__main__":
    main()
