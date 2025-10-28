# ReWire ML App
ReWire ML App is a FastAPI-based machine learning application that processes questionnaire inputs, applies a Scikit-learn preprocessing pipeline, and ranks stimuli using a pre-trained ONNX model. It includes ONNX inference, feature transformation, stimulus mapping, and CSV logging with a minimal HTML interface. The app is lightweight and deployment-ready.

## Features
- FastAPI backend with simple UI
- ONNX model inference pipeline
- Joblib preprocessing transformer
- Top-N stimulus ranking logic
- CSV-based logs
- Render deployable

## Tech Stack
Python • FastAPI • ONNX Runtime • Scikit-learn • Jinja2 • Uvicorn

## Local Setup
git clone https://github.com/Ashwin0410/rewire-ml-app.git
cd rewire-ml-app
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Visit: http://localhost:8000

## Deployment
Configured for Render using:
- Procfile
- runtime.txt
- requirements.txt

## Project Structure
rewire-ml-app/
│ app.py
│ final_global_mlp.onnx
│ preprocessor_minimal.joblib
│ stimuli_mapping.json
│ logs.csv
│ requirements.txt
│ Procfile
│ runtime.txt
└── templates/

