from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "predictive_maintenance_model.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = float(artifact["threshold"])

def predict(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    proba = float(model.predict_proba(data)[:, 1][0])
    prediction = int(proba >= threshold)
    return {
        "failure_prediction": prediction,
        "failure_probability": proba,
        "threshold": threshold,
        "status": "Failure Risk" if prediction else "Normal"
    }
