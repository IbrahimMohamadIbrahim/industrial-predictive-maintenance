"""Load and apply the versioned predictive-maintenance model artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "models" / "predictive_maintenance_model.joblib",
    ROOT / "notebooks" / "artifacts" / "predictive_maintenance_model.joblib",
)


class Predictor:
    """Thin, validated interface around the saved scikit-learn pipeline."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        artifact_path = artifact_path or next((path for path in MODEL_PATHS if path.is_file()), None)
        if artifact_path is None:
            expected = ", ".join(str(path) for path in MODEL_PATHS)
            raise FileNotFoundError(f"Model artifact not found. Expected one of: {expected}")
        artifact = joblib.load(artifact_path)
        required = {"model", "threshold", "features", "target", "model_name"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.features = list(artifact["features"])
        self.target = str(artifact["target"])
        self.model_name = str(artifact["model_name"])

    def predict(self, data: Mapping[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Return a probability and decision using the artifact's saved threshold."""
        frame = pd.DataFrame([data]) if isinstance(data, Mapping) else data.copy()
        missing = [feature for feature in self.features if feature not in frame.columns]
        if missing:
            raise ValueError(f"Prediction inputs are missing required features: {missing}")
        features = frame.loc[:, self.features]
        probability = float(self.model.predict_proba(features)[:, 1][0])
        prediction = int(probability >= self.threshold)
        return {
            "failure_prediction": prediction,
            "failure_probability": probability,
            "threshold": self.threshold,
            "status": "Failure Risk" if prediction else "Normal",
            "model_name": self.model_name,
            "features": self.features,
        }
