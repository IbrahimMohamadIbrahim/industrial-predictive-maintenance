"""Read-only Deep Learning handoff validation for the deployment dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEEP_LEARNING_DIR = ROOT / "models" / "deep_learning"
MANIFEST_PATH = DEEP_LEARNING_DIR / "model_manifest.json"


@dataclass(frozen=True)
class ModelHandoffStatus:
    """Readiness information for one DL model handed to the deployment team."""

    name: str
    ready: bool
    messages: tuple[str, ...]
    metadata: dict[str, Any] | None = None


def _safe_artifact_path(relative_path: str) -> Path | None:
    """Resolve a manifest artifact only when it remains inside the DL model folder."""
    candidate = (DEEP_LEARNING_DIR / relative_path).resolve()
    try:
        candidate.relative_to(DEEP_LEARNING_DIR.resolve())
    except ValueError:
        return None
    return candidate


def _check_model(name: str, metadata: Any) -> ModelHandoffStatus:
    if not isinstance(metadata, dict):
        return ModelHandoffStatus(
            name=name,
            ready=False,
            messages=("No metadata has been supplied yet.",),
        )

    required = ["framework", "artifact", "preprocessing_artifact", "feature_order"]
    if name == "MLP":
        required.append("decision_threshold")
    else:
        required.extend(["anomaly_threshold", "score_method"])

    messages = [f"Missing required metadata: `{field}`." for field in required if field not in metadata]
    feature_order = metadata.get("feature_order")
    if "feature_order" in metadata and (
        not isinstance(feature_order, list) or not all(isinstance(item, str) for item in feature_order)
    ):
        messages.append("`feature_order` must be a list of feature-name strings.")

    for field in ("artifact", "preprocessing_artifact"):
        relative_path = metadata.get(field)
        if not isinstance(relative_path, str) or not relative_path.strip():
            continue
        artifact_path = _safe_artifact_path(relative_path)
        if artifact_path is None:
            messages.append(f"`{field}` must stay inside `models/deep_learning`.")
        elif not artifact_path.is_file():
            messages.append(f"Expected file is missing: `{relative_path}`.")

    return ModelHandoffStatus(
        name=name,
        ready=not messages,
        messages=tuple(messages),
        metadata=metadata,
    )


def load_deep_learning_handoff(path: Path = MANIFEST_PATH) -> tuple[ModelHandoffStatus, ModelHandoffStatus]:
    """Load the optional DL manifest without loading or executing model files."""
    if not path.is_file():
        waiting = "Awaiting `models/deep_learning/model_manifest.json` from the Deep Learning teammate."
        return (
            ModelHandoffStatus(name="MLP", ready=False, messages=(waiting,)),
            ModelHandoffStatus(name="Autoencoder", ready=False, messages=(waiting,)),
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"Cannot read the Deep Learning manifest: {type(exc).__name__}."
        return (
            ModelHandoffStatus(name="MLP", ready=False, messages=(message,)),
            ModelHandoffStatus(name="Autoencoder", ready=False, messages=(message,)),
        )

    if not isinstance(manifest, dict):
        message = "The Deep Learning manifest must be a JSON object."
        return (
            ModelHandoffStatus(name="MLP", ready=False, messages=(message,)),
            ModelHandoffStatus(name="Autoencoder", ready=False, messages=(message,)),
        )

    return _check_model("MLP", manifest.get("mlp")), _check_model("Autoencoder", manifest.get("autoencoder"))
