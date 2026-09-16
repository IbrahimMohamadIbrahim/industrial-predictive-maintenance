"""Tests for app/dl_inference.py: the layer that actually loads and runs the
MLP and autoencoder, on top of deep_learning.py's file-existence checks.

These build tiny throwaway Keras models in a temp directory so the tests do
not depend on the real Deep Learning teammate's artifacts ever existing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import deep_learning  # noqa: E402
import dl_inference  # noqa: E402

MLP_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AE_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

SAMPLE_INPUT = {
    "Type": "M",
    "Air temperature [K]": 302.0,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1400.0,
    "Torque [Nm]": 55.0,
    "Tool wear [min]": 210.0,
}


@pytest.fixture
def dl_dir(tmp_path, monkeypatch) -> Path:
    """Point both deep_learning.py and dl_inference.py at a scratch directory."""
    scratch = tmp_path / "models" / "deep_learning"
    (scratch / "mlp").mkdir(parents=True)
    (scratch / "autoencoder").mkdir(parents=True)
    monkeypatch.setattr(deep_learning, "DEEP_LEARNING_DIR", scratch)
    monkeypatch.setattr(deep_learning, "MANIFEST_PATH", scratch / "model_manifest.json")
    monkeypatch.setattr(dl_inference, "DEEP_LEARNING_DIR", scratch)
    return scratch


def _write_valid_artifacts(dl_dir: Path) -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from tensorflow import keras

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 2, n),
            "Rotational speed [rpm]": rng.normal(1500, 150, n),
            "Torque [Nm]": rng.normal(40, 8, n),
            "Tool wear [min]": rng.uniform(0, 250, n),
        }
    )
    y = rng.integers(0, 2, size=n)

    mlp_pre = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
            ),
            ("cat", OneHotEncoder(categories=[["L", "M", "H"]], handle_unknown="ignore"), ["Type"]),
        ]
    )
    x_mlp = np.asarray(mlp_pre.fit_transform(df[MLP_FEATURES]), dtype="float32")
    mlp_model = keras.Sequential(
        [keras.layers.Input(shape=(x_mlp.shape[1],)), keras.layers.Dense(4, activation="relu"), keras.layers.Dense(1, activation="sigmoid")]
    )
    mlp_model.compile(optimizer="adam", loss="binary_crossentropy")
    mlp_model.fit(x_mlp, y, epochs=1, verbose=0)
    mlp_model.save(dl_dir / "mlp" / "model.keras")
    joblib.dump(mlp_pre, dl_dir / "mlp" / "preprocessing.joblib")

    ae_pre = StandardScaler()
    x_ae = ae_pre.fit_transform(df[AE_FEATURES]).astype("float32")
    inputs = keras.layers.Input(shape=(x_ae.shape[1],))
    encoded = keras.layers.Dense(2, activation="relu")(inputs)
    decoded = keras.layers.Dense(x_ae.shape[1], activation="linear")(encoded)
    ae_model = keras.Model(inputs, decoded)
    ae_model.compile(optimizer="adam", loss="mse")
    ae_model.fit(x_ae, x_ae, epochs=1, verbose=0)
    ae_model.save(dl_dir / "autoencoder" / "model.keras")
    joblib.dump(ae_pre, dl_dir / "autoencoder" / "preprocessing.joblib")

    manifest = {
        "schema_version": "1.0",
        "mlp": {
            "framework": "tensorflow",
            "framework_version": "test",
            "artifact": "mlp/model.keras",
            "preprocessing_artifact": "mlp/preprocessing.joblib",
            "feature_order": MLP_FEATURES,
            "decision_threshold": 0.5,
        },
        "autoencoder": {
            "framework": "tensorflow",
            "framework_version": "test",
            "artifact": "autoencoder/model.keras",
            "preprocessing_artifact": "autoencoder/preprocessing.joblib",
            "feature_order": AE_FEATURES,
            "score_method": "mean_squared_reconstruction_error",
            "anomaly_threshold": 1.0,
        },
    }
    (dl_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_no_manifest_is_not_ready(dl_dir):
    mlp_rt, ae_rt = dl_inference.load_dl_runtime()
    assert mlp_rt.ready is False
    assert ae_rt.ready is False
    assert mlp_rt.model is None


def test_valid_artifacts_load_and_infer(dl_dir):
    _write_valid_artifacts(dl_dir)
    mlp_rt, ae_rt = dl_inference.load_dl_runtime()
    assert mlp_rt.ready is True
    assert ae_rt.ready is True

    mlp_result = dl_inference.predict_mlp(mlp_rt.model, SAMPLE_INPUT)
    assert 0.0 <= mlp_result["failure_probability"] <= 1.0
    assert mlp_result["failure_prediction"] in (0, 1)
    assert mlp_result["threshold"] == pytest.approx(0.5)

    ae_result = dl_inference.score_autoencoder(ae_rt.model, SAMPLE_INPUT)
    assert ae_result["reconstruction_error"] >= 0.0
    assert ae_result["is_anomaly"] in (True, False)


def test_missing_feature_raises_dl_inference_error(dl_dir):
    _write_valid_artifacts(dl_dir)
    mlp_rt, _ = dl_inference.load_dl_runtime()
    incomplete_input = {"Type": "M", "Air temperature [K]": 300.0}
    with pytest.raises(dl_inference.DLInferenceError):
        dl_inference.predict_mlp(mlp_rt.model, incomplete_input)


def test_corrupted_model_file_is_reported_not_ready(dl_dir):
    _write_valid_artifacts(dl_dir)
    (dl_dir / "mlp" / "model.keras").write_text("not a real keras archive", encoding="utf-8")
    mlp_rt, ae_rt = dl_inference.load_dl_runtime()
    assert mlp_rt.ready is False
    assert "could not be loaded" in mlp_rt.messages[0]
    # Autoencoder is untouched and should still load fine.
    assert ae_rt.ready is True