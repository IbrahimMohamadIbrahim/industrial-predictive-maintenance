from pathlib import Path
import sqlite3

import pandas as pd

from scripts.build_database import build_database
from scripts.run_sql_queries import read_sql_statements
from src.data.clean_data import clean_data
from src.data.load_data import EXPECTED_COLUMNS, load_raw_data, save_processed_data
from src.data.validate_data import run_validation


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, "M14860", "M", 298.1, 308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0, 0],
            [2, "L47181", "L", 298.2, 308.7, 1408, 46.3, 3, 0, 0, 0, 0, 0, 0],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_loading_and_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "ai4i2020.csv"
    sample_data().to_csv(path, index=False)
    loaded = load_raw_data(path)
    assert not loaded.empty
    assert list(loaded.columns) == EXPECTED_COLUMNS


def test_cleaning_normalizes_categories_and_removes_duplicates() -> None:
    raw = sample_data()
    raw.loc[0, "Type"] = " m "
    raw.loc[0, "Product ID"] = " m14860 "
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(raw)

    assert len(cleaned) == 2
    assert cleaned.loc[0, "Type"] == "M"
    assert cleaned.loc[0, "Product ID"] == "M14860"


def test_validation_catches_invalid_target_and_impossible_value() -> None:
    invalid = sample_data().copy()
    invalid.loc[0, "Machine failure"] = 2
    invalid.loc[1, "Torque [Nm]"] = -1

    report = run_validation(invalid)

    assert not report["passed"]
    assert any("Invalid target values" in error for error in report["errors"])
    assert any("negative" in error for error in report["errors"])


def test_cleaning_does_not_hide_invalid_rows() -> None:
    invalid = sample_data().copy()
    invalid.loc[0, "Rotational speed [rpm]"] = -10
    cleaned = clean_data(invalid)
    assert len(cleaned) == len(invalid)
    assert not run_validation(cleaned)["passed"]


def test_processed_data_can_be_saved(tmp_path: Path) -> None:
    output = tmp_path / "processed_data.csv"
    path = save_processed_data(clean_data(sample_data()), output)
    assert path.exists()
    assert not pd.read_csv(path).empty


def test_database_schema_and_all_sql_queries_execute(tmp_path: Path) -> None:
    # Use the real schema but a tiny valid fixture; row-count mismatch is only a warning.
    csv_path = tmp_path / "processed_data.csv"
    db_path = tmp_path / "test.sqlite3"
    clean_data(sample_data()).to_csv(csv_path, index=False)
    build_database(csv_path, db_path)

    root = Path(__file__).resolve().parents[1]
    statements = read_sql_statements(root / "sql" / "queries.sql")
    assert len(statements) >= 10

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        assert connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
        for statement in statements:
            connection.execute(statement).fetchall()
