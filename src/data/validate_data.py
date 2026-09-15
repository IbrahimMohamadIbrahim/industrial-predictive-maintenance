"""Validation checks for the AI4I 2020 data pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .clean_data import BINARY_COLUMNS, NUMERIC_COLUMNS
from .load_data import EXPECTED_COLUMNS

EXPECTED_ROW_COUNT = 10_000
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def validate_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Validate required columns and report non-critical extras."""
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    errors = [f"Missing expected columns: {missing}"] if missing else []
    warnings = [f"Unexpected extra columns retained: {extra}"] if extra else []
    return errors, warnings


def validate_missing_values(df: pd.DataFrame, max_missing_pct: float = 0.0) -> list[str]:
    """Require missing-value percentages not to exceed a chosen threshold."""
    if df.empty:
        return ["Dataset is empty."]
    percentages = df.isna().mean() * 100
    return [
        f"Column '{column}' has {percentage:.2f}% missing values "
        f"(allowed: <= {max_missing_pct:.2f}%)."
        for column, percentage in percentages.items()
        if percentage > max_missing_pct
    ]


def validate_duplicates(df: pd.DataFrame) -> list[str]:
    """Check whole-row duplicates and identifier uniqueness."""
    errors: list[str] = []
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        errors.append(f"Dataset contains {duplicate_rows} duplicate rows.")

    for identifier in ["UDI", "Product ID"]:
        if identifier in df.columns:
            duplicate_ids = int(df[identifier].duplicated().sum())
            if duplicate_ids:
                errors.append(f"Column '{identifier}' contains {duplicate_ids} duplicate identifiers.")
    return errors


def validate_target(df: pd.DataFrame) -> list[str]:
    """Validate binary target and failure-mode label domains."""
    errors: list[str] = []
    if TARGET_COLUMN not in df.columns:
        return [f"Target column '{TARGET_COLUMN}' is missing."]

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue
        values = set(df[column].dropna().unique().tolist())
        invalid = sorted(values.difference({0, 1}))
        if invalid:
            prefix = "Invalid target values" if column == TARGET_COLUMN else f"Invalid values in {column}"
            errors.append(f"{prefix}: {invalid}")
    return errors


def validate_numeric_types(df: pd.DataFrame) -> list[str]:
    """Ensure expected numeric columns use numeric dtypes."""
    return [
        f"Column '{column}' is not numeric."
        for column in NUMERIC_COLUMNS
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])
    ]


def validate_ranges(df: pd.DataFrame) -> list[str]:
    """Check documented categorical domains and physically sensible values."""
    errors: list[str] = []
    if "UDI" in df.columns and not df["UDI"].between(1, EXPECTED_ROW_COUNT).all():
        errors.append(f"UDI values must be between 1 and {EXPECTED_ROW_COUNT} for the official dataset.")
    if "Product ID" in df.columns and not df["Product ID"].astype("string").str.match(
        r"^[LMH]\d+$", na=False
    ).all():
        errors.append("Product ID contains values outside the documented L/M/H + serial-number format.")
    if "Type" in df.columns and not df["Type"].isin(["L", "M", "H"]).all():
        errors.append("Type contains values outside the expected categories L/M/H.")

    positive_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
    ]
    for column in positive_columns:
        if column in df.columns and not (df[column] > 0).all():
            errors.append(f"{column} contains non-positive values.")
    if "Torque [Nm]" in df.columns and not (df["Torque [Nm]"] >= 0).all():
        errors.append("Torque [Nm] contains negative values.")
    if "Tool wear [min]" in df.columns and not (df["Tool wear [min]"] >= 0).all():
        errors.append("Tool wear [min] contains negative values.")
    return errors


def validate_identifier_consistency(df: pd.DataFrame) -> list[str]:
    """Check that Product ID's quality prefix agrees with the Type column."""
    if "Product ID" not in df.columns or "Type" not in df.columns:
        return []
    product_prefix = df["Product ID"].astype("string").str[0]
    mismatch_count = int((product_prefix != df["Type"]).fillna(True).sum())
    return [
        f"{mismatch_count} rows have a Product ID prefix that does not match Type."
    ] if mismatch_count else []


def validation_warnings(df: pd.DataFrame) -> list[str]:
    """Return informative warnings that should not invalidate published data."""
    warnings: list[str] = []
    if len(df) != EXPECTED_ROW_COUNT:
        warnings.append(
            f"Row count is {len(df):,}; the official AI4I file contains {EXPECTED_ROW_COUNT:,} rows."
        )

    if all(column in df.columns for column in [TARGET_COLUMN, *FAILURE_MODE_COLUMNS]):
        any_mode = df[FAILURE_MODE_COLUMNS].max(axis=1)
        mismatch_count = int((df[TARGET_COLUMN] != any_mode).sum())
        if mismatch_count:
            warnings.append(
                f"{mismatch_count} published rows have Machine failure different from the OR of the five "
                "failure-mode flags. These labels are preserved as source data, not rewritten by the pipeline."
            )
    return warnings


def run_validation(df: pd.DataFrame, max_missing_pct: float = 0.0) -> dict[str, Any]:
    """Run all validation checks and return a structured pass/fail report."""
    errors: list[str] = []
    warnings: list[str] = []

    schema_errors, schema_warnings = validate_schema(df)
    errors.extend(schema_errors)
    warnings.extend(schema_warnings)

    if df.empty:
        errors.append("Dataset is empty.")
    else:
        errors.extend(validate_missing_values(df, max_missing_pct=max_missing_pct))
        errors.extend(validate_duplicates(df))
        errors.extend(validate_target(df))
        errors.extend(validate_numeric_types(df))
        errors.extend(validate_ranges(df))
        errors.extend(validate_identifier_consistency(df))
        warnings.extend(validation_warnings(df))

    return {
        "passed": not errors,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "errors": errors,
        "warnings": warnings,
    }
