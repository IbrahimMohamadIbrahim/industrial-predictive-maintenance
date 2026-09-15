# Data layer

## Dataset and source

The official project brief specifies the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository (dataset 601). It is a synthetic predictive-maintenance benchmark with 10,000 rows. The original dataset creators/source should be cited in the final report.

Official UCI page: <https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset>

Raw file expected by the pipeline:

```text
data/raw/ai4i2020.csv
```

Generated cleaned/validated output:

```text
data/processed/processed_data.csv
```

The repository `.gitignore` excludes raw CSV files and generated processed artifacts so external data is not accidentally committed. The local project copy may still contain them for execution/testing.

## Data dictionary

| Column | Role in source | Meaning / unit | Task 1 treatment |
|---|---|---|---|
| `UDI` | Identifier | Unique row identifier, 1-10000 | Preserved; validate uniqueness; do not treat as a predictive sensor feature by default. |
| `Product ID` | Identifier | Product identifier beginning with L/M/H | Preserved; whitespace/case normalized; validate uniqueness and prefix. |
| `Type` | Feature | Product quality variant: L, M, H | Preserved; whitespace/case normalized. |
| `Air temperature [K]` | Feature | Air temperature in kelvin | Numeric coercion + positive-value validation. |
| `Process temperature [K]` | Feature | Process temperature in kelvin | Numeric coercion + positive-value validation. |
| `Rotational speed [rpm]` | Feature | Rotational speed in rpm | Numeric coercion + positive-value validation. |
| `Torque [Nm]` | Feature | Torque in newton-metres | Numeric coercion + non-negative validation. |
| `Tool wear [min]` | Feature | Tool-wear duration in minutes | Numeric coercion + non-negative validation. |
| `Machine failure` | Target | Overall binary failure label | Preserved as 0/1; never used to clean/construct predictors. |
| `TWF` | Target/label | Tool-wear failure flag | Preserved for failure-mode analysis; **exclude as a predictor of `Machine failure`**. |
| `HDF` | Target/label | Heat-dissipation failure flag | Same leakage warning as above. |
| `PWF` | Target/label | Power failure flag | Same leakage warning as above. |
| `OSF` | Target/label | Overstrain failure flag | Same leakage warning as above. |
| `RNF` | Target/label | Random failure flag | Same leakage warning as above. |

## Cleaning performed

The reusable cleaning code is intentionally conservative and leakage-safe. It trims column names, removes accidental `Unnamed:*` CSV index columns, normalizes `Product ID`/`Type` case and whitespace, coerces documented numeric fields to numeric dtypes, and removes exact duplicate rows. It does **not** impute, clip, scale, encode, feature-engineer, or remove rows because of target values.

The official raw file currently supplied with the project has 10,000 rows, 14 expected columns, no missing values, and no exact duplicate rows, so no source records are discarded by the pipeline.

## Validation behavior

Validation checks required columns, missingness, row/identifier duplicates, numeric dtypes, binary target/label domains, L/M/H categories, Product ID/Type consistency, and physically impossible negative/non-positive measurements. A row-count difference from 10,000 is reported as a warning rather than an automatic error so unit-test fixtures and explicitly sampled data can still be validated.

The published AI4I labels contain a small number of rows where `Machine failure` does not equal the simple OR of all five failure-mode flags. Task 1 reports this as a warning and preserves the source labels instead of silently rewriting them.

## Leakage handoff

For downstream **binary `Machine failure` prediction**, do not use `TWF`, `HDF`, `PWF`, `OSF`, or `RNF` as input features: UCI defines them as failure labels/targets, so using them would leak outcome information. Model-specific splitting, scaling, encoding, resampling, and feature engineering belong to later team tasks, not this data-cleaning stage.
