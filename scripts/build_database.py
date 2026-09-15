"""Build a validated SQLite database from processed AI4I data."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validate_data import run_validation

SCHEMA_PATH = ROOT / "sql" / "schema.sql"
DEFAULT_CSV = ROOT / "data" / "processed" / "processed_data.csv"
DEFAULT_DB = ROOT / "data" / "processed" / "predictive_maintenance.sqlite3"


def build_database(csv_path: Path = DEFAULT_CSV, db_path: Path = DEFAULT_DB) -> Path:
    """Create and populate the relational SQLite database from processed CSV data."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}. Run the data pipeline first.")

    dataframe = pd.read_csv(csv_path)
    validation = run_validation(dataframe)
    if not validation["passed"]:
        raise ValueError(f"Processed CSV failed validation: {validation['errors']}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        products = dataframe[["Product ID", "Type"]].drop_duplicates().rename(
            columns={"Product ID": "product_id", "Type": "product_type"}
        )
        products.to_sql("products", connection, if_exists="append", index=False)

        observations = dataframe[
            [
                "UDI",
                "Product ID",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]
        ].rename(
            columns={
                "UDI": "udi",
                "Product ID": "product_id",
                "Air temperature [K]": "air_temperature_k",
                "Process temperature [K]": "process_temperature_k",
                "Rotational speed [rpm]": "rotational_speed_rpm",
                "Torque [Nm]": "torque_nm",
                "Tool wear [min]": "tool_wear_min",
            }
        )
        observations.to_sql("observations", connection, if_exists="append", index=False)

        failures = dataframe[
            ["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        ].rename(
            columns={
                "UDI": "udi",
                "Machine failure": "machine_failure",
                "TWF": "twf",
                "HDF": "hdf",
                "PWF": "pwf",
                "OSF": "osf",
                "RNF": "rnf",
            }
        )
        failures.to_sql("failures", connection, if_exists="append", index=False)

        foreign_key_issues = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_issues:
            raise RuntimeError(f"Foreign-key validation failed: {foreign_key_issues[:5]}")

    return db_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    print(f"Created SQLite database: {build_database(args.csv, args.db)}")
