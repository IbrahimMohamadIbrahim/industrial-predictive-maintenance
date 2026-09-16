from __future__ import annotations

import json
from pathlib import Path

from app.deep_learning import load_deep_learning_handoff


def test_missing_manifest_reports_pending_handoff(tmp_path: Path) -> None:
    mlp, autoencoder = load_deep_learning_handoff(tmp_path / "model_manifest.json")

    assert not mlp.ready
    assert not autoencoder.ready
    assert "Awaiting" in mlp.messages[0]


def test_complete_manifest_with_local_artifacts_reports_ready(tmp_path: Path, monkeypatch) -> None:
    deep_learning_dir = tmp_path / "models" / "deep_learning"
    (deep_learning_dir / "mlp").mkdir(parents=True)
    (deep_learning_dir / "autoencoder").mkdir(parents=True)
    for relative_path in (
        "mlp/model.keras",
        "mlp/preprocessing.joblib",
        "autoencoder/model.keras",
        "autoencoder/preprocessing.joblib",
    ):
        (deep_learning_dir / relative_path).touch()

    manifest = {
        "mlp": {
            "framework": "tensorflow",
            "artifact": "mlp/model.keras",
            "preprocessing_artifact": "mlp/preprocessing.joblib",
            "feature_order": ["Torque [Nm]"],
            "decision_threshold": 0.5,
        },
        "autoencoder": {
            "framework": "tensorflow",
            "artifact": "autoencoder/model.keras",
            "preprocessing_artifact": "autoencoder/preprocessing.joblib",
            "feature_order": ["Torque [Nm]"],
            "score_method": "mse",
            "anomaly_threshold": 0.2,
        },
    }
    manifest_path = deep_learning_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("app.deep_learning.DEEP_LEARNING_DIR", deep_learning_dir)

    mlp, autoencoder = load_deep_learning_handoff(manifest_path)

    assert mlp.ready
    assert autoencoder.ready


def test_manifest_rejects_artifacts_outside_deep_learning_folder(tmp_path: Path, monkeypatch) -> None:
    deep_learning_dir = tmp_path / "models" / "deep_learning"
    deep_learning_dir.mkdir(parents=True)
    manifest = {
        "mlp": {
            "framework": "tensorflow",
            "artifact": "../../outside.keras",
            "preprocessing_artifact": "../../outside.joblib",
            "feature_order": ["Torque [Nm]"],
            "decision_threshold": 0.5,
        }
    }
    manifest_path = deep_learning_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("app.deep_learning.DEEP_LEARNING_DIR", deep_learning_dir)

    mlp, _ = load_deep_learning_handoff(manifest_path)

    assert not mlp.ready
    assert any("must stay inside" in message for message in mlp.messages)
