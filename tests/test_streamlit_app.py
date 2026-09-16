from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_dashboard_loads_and_scores_a_prediction() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert app.title[0].value == "🏭 Industrial Predictive Maintenance"

    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[0].value.endswith("%")


def test_dashboard_answers_from_local_rag_sources_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_ONLY", "1")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    app.chat_input(key="rag_question").set_value("What does HDF mean in AI4I?").run(timeout=20)

    assert not app.exception
    assert "retrieval-only" in app.markdown[-2].value.lower() or any(
        "retrieval-only" in item.value.lower() for item in app.markdown
    )
    assert any("03_ai4i_failure_modes.md" in item.value for item in app.markdown)
