# ReWire ML App
ReWire ML App is a FastAPI-based machine learning application that processes questionnaire inputs and generates ranked stimuli recommendations using a pre-trained ONNX model and a Scikit-learn preprocessing pipeline. It includes model inference, feature transformation, stimulus mapping, and CSV-based logging with a simple HTML frontend.

## What This App Does
This app takes answers from a small questionnaire, preprocesses them using a saved Scikit-learn pipeline, and sends the data to a trained ONNX model. The model then scores and ranks 40 predefined emotional audio/video stimuli and returns a personalized recommendation list. Everything runs through a FastAPI backend with a clean HTML interface, and all predictions are logged for tracking and improvement.

## Features
- FastAPI backend
- ONNX model inference
- Scikit-learn preprocessing
- Stimulus ranking logic
- CSV logging
- Ready for Render deployment

## Tech Stack
Python • FastAPI • ONNX Runtime • Scikit-learn • Jinja2 • Uvicorn

## Local Setup
git clone https://github.com/Ashwin0410/rewire-ml-app.git

cd rewire-ml-app

pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Open in browser: http://localhost:8000

## Deployment (Render)
Uses:
- Procfile
- runtime.txt
- requirements.txt

## Project Structure
rewire-ml-app/

 app.py
 
 final_global_mlp.onnx
 
 preprocessor_minimal.joblib
 
 stimuli_mapping.json
 
 logs.csv
 
 requirements.txt
 
 Procfile
 
 runtime.txt
 
 templates/

