"""Run the Task 1 AI4I data pipeline from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean_data import clean_data
from src.data.load_data import default_processed_path, default_raw_path, load_raw_data, save_processed_data
from src.data.validate_data import run_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=default_raw_path(), help="Input raw CSV path.")
    parser.add_argument(
        "--output", type=Path, default=default_processed_path(), help="Processed CSV output path."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load, clean, and validate without writing processed_data.csv.",
    )
    args = parser.parse_args()

    raw = load_raw_data(args.raw)
    cleaned = clean_data(raw)
    report = run_validation(cleaned)

    print(f"Raw shape:       {raw.shape}")
    print(f"Cleaned shape:   {cleaned.shape}")
    print(f"Validation pass: {report['passed']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")

    if not report["passed"]:
        return 1
    if not args.validate_only:
        output_path = save_processed_data(cleaned, args.output)
        print(f"Processed data:  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
