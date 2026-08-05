Automated Chest X-Ray Pathology Screening
A complete AI-based medical image diagnostics project for pneumonia screening using deep learning and explainable AI.

Project overview
This repository includes a full pipeline for chest X-ray classification with:

Dataset preprocessing
CNN training with EfficientNetB0 transfer learning
Single-image prediction
Grad-CAM explainability
Radiology report generation
Streamlit dashboard for local usage
Folder structure
Chest-Xray-Diagnosis
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── chest_xray_model.h5
│
├── outputs/
│   ├── heatmaps/
│   ├── reports/
│   ├── csv/
│   └── uploads/
│
├── preprocess.py
├── train.py
├── predict.py
├── gradcam.py
├── report.py
├── app.py
├── utils.py
├── requirements.txt
└── README.md
Installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Dataset
Place the Kaggle Chest X-Ray dataset inside the dataset/archive/chest_xray/chest_xray directory, or use the existing split folders under dataset/train, dataset/val, and dataset/test.
Preprocessing
python preprocess.py
This script resizes images to 224x224, normalizes pixel values, and saves processed data under outputs/processed.
Model training
python train.py
The script trains a CNN with EfficientNetB0 transfer learning, saves the model to models/chest_xray_model.h5, and generates training charts and evaluation reports.
Prediction
python predict.py
This script loads the saved model and predicts the first test chest X-ray image. It saves a CSV record and audit log.
Grad-CAM
python gradcam.py
This script generates a Grad-CAM heatmap overlay for the sample test image and saves it to outputs/heatmaps/.
Report generation
python report.py
Generates a PDF radiology report and a summary CSV in outputs/reports/ and outputs/csv/.
Streamlit dashboard
streamlit run app.py
This launches a local dashboard for uploading chest X-ray images, viewing predictions, and exporting results.
Future improvements
Add multi-class disease classification beyond pneumonia
Improve model generalization with more data augmentation
Add DICOM support and PACS integration
Use a stronger explainability pipeline with Grad-CAM++
Deploy as a web service using Docker or Azure App Service
