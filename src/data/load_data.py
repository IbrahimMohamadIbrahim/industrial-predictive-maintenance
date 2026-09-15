"""Loading and persistence helpers for the AI4I 2020 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
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


def project_root() -> Path:
    """Return the repository root using this module's location."""
    return Path(__file__).resolve().parents[2]


def default_raw_path() -> Path:
    """Return the expected project-relative raw CSV path."""
    return project_root() / "data" / "raw" / "ai4i2020.csv"


def default_processed_path() -> Path:
    """Return the default project-relative processed CSV path."""
    return project_root() / "data" / "processed" / "processed_data.csv"


def load_raw_data(path: PathLike | None = None) -> pd.DataFrame:
    """Load the raw AI4I CSV without applying preprocessing.

    Parameters
    ----------
    path:
        Optional CSV path. If omitted, ``data/raw/ai4i2020.csv`` is used.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If the CSV exists but contains no data.
    RuntimeError
        If pandas cannot parse/read the file.
    """
    csv_path = Path(path).expanduser() if path is not None else default_raw_path()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. Download the official "
            "AI4I 2020 CSV from UCI and place it under data/raw/ai4i2020.csv."
        )

    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw dataset is empty: {csv_path}") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read CSV '{csv_path}': {exc}") from exc

    if dataframe.empty:
        raise ValueError(f"Raw dataset contains no rows: {csv_path}")
    return dataframe


def load_processed_data(path: PathLike | None = None) -> pd.DataFrame:
    """Load the generated processed dataset."""
    csv_path = Path(path).expanduser() if path is not None else default_processed_path()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at '{csv_path}'. Run the data pipeline first."
        )
    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Processed dataset is empty: {csv_path}") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read CSV '{csv_path}': {exc}") from exc
    if dataframe.empty:
        raise ValueError(f"Processed dataset contains no rows: {csv_path}")
    return dataframe


def save_processed_data(df: pd.DataFrame, path: PathLike | None = None) -> Path:
    """Save a processed DataFrame as CSV and return the output path."""
    if df.empty:
        raise ValueError("Refusing to save an empty processed dataset.")

    output_path = Path(path).expanduser() if path is not None else default_processed_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Could not write processed CSV '{output_path}': {exc}") from exc
    return output_path
