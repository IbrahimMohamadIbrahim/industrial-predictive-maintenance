"""Load and run the Deep Learning teammate's MLP and autoencoder artifacts.

`deep_learning.py` only checks that the manifest is well-formed and that the
artifact files exist on disk -- it deliberately never loads or executes them
(see its module docstring). This module is the next layer: it takes a
manifest entry that already passed that check, actually loads the `.keras`
model and its `preprocessing.joblib` companion, and runs real inference.

Design contract assumed for `preprocessing_artifact` (confirm with the Deep
Learning teammate): a fitted scikit-learn-compatible object with
`.transform(DataFrame)` that accepts the manifest's `feature_order` columns,
in order, and returns an array ready to feed directly to the Keras model.
This mirrors the existing classical-model contract in `model.py`.

Nothing here changes `deep_learning.py`. If loading or inference fails for
any reason (corrupt file, shape mismatch, wrong preprocessing contract), the
caller gets a clear `DLInferenceError` instead of a crash or a silent wrong
answer -- consistent with "do not say the model is integrated until it can
perform inference."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from deep_learning import DEEP_LEARNING_DIR, ModelHandoffStatus, load_deep_learning_handoff


class DLInferenceError(Exception):
    """Raised when a DL artifact cannot be loaded or run, even though the
    handoff-validation check reported it as ready."""


@dataclass(frozen=True)
class LoadedDLModel:
    """A DL model actually loaded into memory, ready for inference."""

    name: str
    keras_model: Any
    preprocessing: Any
    feature_order: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DLRuntime:
    """End result of attempting to make a manifest entry actually runnable."""

    name: str
    ready: bool
    messages: tuple[str, ...]
    model: LoadedDLModel | None = None


def _load_one(status: ModelHandoffStatus) -> DLRuntime:
    """Attempt to actually load a model that already passed handoff validation."""
    if not status.ready or status.metadata is None:
        return DLRuntime(name=status.name, ready=False, messages=status.messages)

    metadata = status.metadata
    try:
        # Imported lazily: importing TensorFlow is slow (a few seconds) and
        # unnecessary for every Streamlit rerun where the handoff isn't ready.
        from tensorflow import keras
    except ImportError as exc:
        return DLRuntime(
            name=status.name,
            ready=False,
            messages=(f"TensorFlow is not installed, cannot load `{status.name}`: {exc}",),
        )

    artifact_path = DEEP_LEARNING_DIR / metadata["artifact"]
    preprocessing_path = DEEP_LEARNING_DIR / metadata["preprocessing_artifact"]

    try:
        keras_model = keras.models.load_model(artifact_path)
    except Exception as exc:  # keras/TF can raise many different error types
        return DLRuntime(
            name=status.name,
            ready=False,
            messages=(f"`{metadata['artifact']}` exists but could not be loaded: {exc}",),
        )

    try:
        preprocessing = joblib.load(preprocessing_path)
    except Exception as exc:
        return DLRuntime(
            name=status.name,
            ready=False,
            messages=(f"`{metadata['preprocessing_artifact']}` exists but could not be loaded: {exc}",),
        )

    if not hasattr(preprocessing, "transform"):
        return DLRuntime(
            name=status.name,
            ready=False,
            messages=(
                f"`{metadata['preprocessing_artifact']}` does not expose `.transform()`. "
                "Confirm the preprocessing-artifact contract with the Deep Learning teammate.",
            ),
        )

    loaded = LoadedDLModel(
        name=status.name,
        keras_model=keras_model,
        preprocessing=preprocessing,
        feature_order=tuple(metadata["feature_order"]),
        metadata=metadata,
    )
    return DLRuntime(name=status.name, ready=True, messages=(), model=loaded)


def load_dl_runtime(path: Path | None = None) -> tuple[DLRuntime, DLRuntime]:
    """Validate the handoff, then actually try to load both models.

    Safe to call even when the manifest or artifacts are missing: returns a
    not-ready `DLRuntime` with a human-readable reason instead of raising.
    """
    if path is None:
        mlp_status, ae_status = load_deep_learning_handoff()
    else:
        mlp_status, ae_status = load_deep_learning_handoff(path)
    return _load_one(mlp_status), _load_one(ae_status)


def _transform(loaded: LoadedDLModel, inputs: Mapping[str, Any]) -> np.ndarray:
    missing = [f for f in loaded.feature_order if f not in inputs]
    if missing:
        raise DLInferenceError(f"Inputs are missing required features: {missing}")
    frame = pd.DataFrame([{f: inputs[f] for f in loaded.feature_order}])
    try:
        transformed = loaded.preprocessing.transform(frame)
    except Exception as exc:
        raise DLInferenceError(f"Preprocessing failed for `{loaded.name}`: {exc}") from exc
    if hasattr(transformed, "toarray"):  # sparse matrix (e.g. OneHotEncoder)
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype="float32")


def predict_mlp(loaded: LoadedDLModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Run the MLP and return a probability + decision, using the manifest threshold."""
    features = _transform(loaded, inputs)
    try:
        raw = np.asarray(loaded.keras_model.predict(features, verbose=0))
    except Exception as exc:
        raise DLInferenceError(f"MLP inference failed: {exc}") from exc

    if raw.ndim == 2 and raw.shape[1] == 1:
        probability = float(raw[0, 0])
    elif raw.ndim == 2 and raw.shape[1] == 2:
        probability = float(raw[0, 1])
    elif raw.ndim == 1:
        probability = float(raw[0])
    else:
        raise DLInferenceError(f"Unexpected MLP output shape: {raw.shape}")

    threshold = float(loaded.metadata["decision_threshold"])
    prediction = int(probability >= threshold)
    return {
        "failure_probability": probability,
        "failure_prediction": prediction,
        "threshold": threshold,
        "status": "Failure Risk" if prediction else "Normal",
        "model_name": "MLP (Deep Learning)",
    }


def score_autoencoder(loaded: LoadedDLModel, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Run the autoencoder and return a reconstruction-error anomaly score.

    This is an anomaly score, not a failure probability -- keep it visually
    and semantically separate from the MLP/classical-model outputs.
    """
    features = _transform(loaded, inputs)
    try:
        reconstruction = np.asarray(loaded.keras_model.predict(features, verbose=0))
    except Exception as exc:
        raise DLInferenceError(f"Autoencoder inference failed: {exc}") from exc

    if reconstruction.shape != features.shape:
        raise DLInferenceError(
            f"Autoencoder output shape {reconstruction.shape} does not match input shape {features.shape}."
        )
    reconstruction_error = float(np.mean((features - reconstruction) ** 2))

    anomaly_threshold = loaded.metadata.get("anomaly_threshold")
    is_anomaly = (
        None if anomaly_threshold is None else bool(reconstruction_error >= float(anomaly_threshold))
    )
    return {
        "reconstruction_error": reconstruction_error,
        "anomaly_threshold": anomaly_threshold,
        "is_anomaly": is_anomaly,
        "score_method": loaded.metadata.get("score_method", "mean_squared_reconstruction_error"),
    }