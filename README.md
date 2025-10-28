# ReWire ML App

ReWire ML App is a FastAPI-based machine learning application that processes questionnaire inputs and ranks stimuli using a pre-trained ONNX model and Scikit-learn preprocessing. It includes ONNX inference, feature transformation, stimulus ranking logic, CSV logging, and is deployment-ready.

---

## Features
- FastAPI backend with HTML interface
- ONNX model inference
- Scikit-learn preprocessing pipeline
- Stimulus ranking and mapping
- CSV-based logs
- Render deployment support

---

## Tech Stack
Python • FastAPI • ONNX Runtime • Scikit-learn • Jinja2 • Uvicorn

---

## Local Setup

```bash
git clone https://github.com/Ashwin0410/rewire-ml-app.git
cd rewire-ml-app
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Open in browser → http://localhost:8000

---

## Deployment (Render)

Uses:
Procfile
runtime.txt
requirements.txt

## Project Structure

rewire-ml-app/
│
├── app.py
├── final_global_mlp.onnx
├── preprocessor_minimal.joblib
├── stimuli_mapping.json
├── logs.csv
├── requirements.txt
├── Procfile
├── runtime.txt
└── templates/
