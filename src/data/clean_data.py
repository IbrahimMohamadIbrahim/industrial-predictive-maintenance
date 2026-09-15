"""Leakage-safe cleaning helpers for the AI4I 2020 dataset."""

from __future__ import annotations

import pandas as pd

from .load_data import EXPECTED_COLUMNS

NUMERIC_COLUMNS = [
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
CATEGORICAL_COLUMNS = ["Product ID", "Type"]
BINARY_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column-name whitespace while preserving official names."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def drop_export_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop accidental CSV index columns such as ``Unnamed: 0``."""
    disposable = [
        column for column in df.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]
    return df.drop(columns=disposable) if disposable else df.copy()


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns and normalize string categories."""
    cleaned = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "Product ID" in cleaned.columns:
        cleaned["Product ID"] = (
            cleaned["Product ID"].astype("string").str.strip().str.upper()
        )
    if "Type" in cleaned.columns:
        cleaned["Type"] = cleaned["Type"].astype("string").str.strip().str.upper()
    return cleaned


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean structural issues without model-specific preprocessing.

    The function intentionally does not impute missing values, clip suspicious
    measurements, scale/encode predictors, or use target values to decide which
    records survive. Invalid values remain visible so the validation layer can
    fail loudly rather than silently hiding data-quality problems.
    """
    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = normalize_column_names(df)
    cleaned = drop_export_index_columns(cleaned)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in cleaned.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    cleaned = coerce_types(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned
