# ReWire ML App

ReWire ML App is a FastAPI-based machine learning application that takes user questionnaire inputs, preprocesses them using a Scikit-learn pipeline, and generates ranked emotional stimulus recommendations using a pre-trained ONNX model. It includes feature engineering, ONNX inference, stimulus mapping, CSV-based logging, and a simple HTML frontend. The project is lightweight, modular, and structured for easy local development, testing, and deployment.

---

### Features
- FastAPI backend with Jinja2 templates
- ONNX model for stimulus ranking
- Joblib preprocessing pipeline
- CSV logging for predictions
- Render deployment compatible

---

### Tech Stack
Python • FastAPI • ONNX Runtime • Scikit-learn • Jinja2 • Uvicorn

---

### Local Setup
```bash
git clone https://github.com/Ashwin0410/rewire-ml-app.git
cd rewire-ml-app
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Visit: http://localhost:8000

### Deployment
Configured for deployment on Render using:
- Procfile
- runtime.txt
- requirements.txt

### Project Structure
app.py                  
templates/              
final_global_mlp.onnx   
preprocessor_minimal.joblib
stimuli_mapping.json
logs.csv
