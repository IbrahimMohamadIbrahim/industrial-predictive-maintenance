# Industrial Predictive Maintenance

This repository is the team project for **Industrial Predictive Maintenance & Failure Prevention**. The work in this local copy focuses on **Task 1: Data Pipeline + SQL** only.

## Task 1 scope

Implemented here:

- reproducible raw -> cleaned -> validated data pipeline;
- processed dataset generation;
- SQLite relational schema and database loader;
- 12 analytical SQL queries, including JOINs, CTEs, aggregates, filtering, grouping, ordering, and window functions;
- Task 1 notebook and basic automated tests;
- dataset/data-dictionary and leakage documentation.

Not implemented in Task 1: EDA/feature engineering, classical ML, deep learning, anomaly detection, SHAP, risk engine, monitoring, Streamlit, or RAG.

## Dataset

The official project requirements specify the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository (dataset 601). It is a synthetic benchmark with 10,000 observations and 14 published columns containing identifiers, product type, operating measurements, an overall machine-failure target, and five failure-mode labels.

Official source: <https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset>

Expected raw-data location:

```text
data/raw/ai4i2020.csv
```

Generated processed-data location:

```text
data/processed/processed_data.csv
```

See `data/README.md` for the data dictionary, cleaning decisions, validation rules, and leakage notes.

### Important leakage note for later ML work

`TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are published failure-mode **target/label columns**. When predicting `Machine failure`, they must not be used as predictor features because that would leak failure information. Task 1 preserves these columns for SQL analysis and possible later failure-mode modeling, but performs no target-derived feature engineering.

## Install dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Task 1 intentionally uses only pandas, pytest, Jupyter, and Python's standard-library `sqlite3`; SQLAlchemy is not required.

## Run the data pipeline

Generate the processed dataset:

```bash
python scripts/run_data_pipeline.py
```

Validate the cleaned data without writing a new processed file:

```bash
python scripts/run_data_pipeline.py --validate-only
```

Custom paths are also supported:

```bash
python scripts/run_data_pipeline.py --raw path/to/ai4i2020.csv --output data/processed/processed_data.csv
```

The cleaning stage is deliberately conservative: it normalizes structural formatting, numeric dtypes, categories, and exact duplicates, but does not impute, scale, encode, clip suspicious values, or drop rows based on target information. Invalid values remain visible to validation so data-quality problems fail loudly.

## Notebook

`notebooks/01_data_pipeline.ipynb` demonstrates the same reusable Python functions used by the pipeline script. It is for explanation/experimentation; later application code must import the Python modules directly rather than depend on notebook execution.

Launch Jupyter:

```bash
jupyter notebook
```

Or execute the notebook non-interactively from the project root:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_pipeline.ipynb \
  --output 01_data_pipeline.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=120
```

The generated `*.executed.ipynb` file is only an execution check and does not need to be committed.

## SQL database

The project brief requires SQL/structured storage but does not prescribe a DBMS, so Task 1 uses **SQLite** for portability and zero server setup.

The relational design is defined in `sql/schema.sql`:

- `products` - Product ID and L/M/H product type;
- `observations` - one operating/sensor observation per UDI;
- `failures` - overall failure target plus the five published failure-mode labels;
- `failure_observation_details` - joined analytical view.

The source dataset does **not** contain a persistent physical-machine ID. Product IDs are unique per observation, so the schema and SQL intentionally avoid claims such as "machine X failed repeatedly" that the data cannot support.

Build the database from validated processed data:

```bash
python scripts/build_database.py
```

Default database output:

```text
data/processed/predictive_maintenance.sqlite3
```

Run and preview every analytical query using Python's built-in SQLite support:

```bash
python scripts/run_sql_queries.py
```

The 12 queries in `sql/queries.sql` cover overall failure prevalence, failure rate by product type, failure-mode frequencies, measurement averages, failed high-wear observations, wear/operating bands, multi-mode failures, ordered UDI blocks, and windowed/ranked failure statistics. Q2/Q4-Q6 use explicit JOINs, Q7/Q8/Q10/Q12 use CTEs, and Q11/Q12 use window functions.

## Tests

Run all Task 1 tests:

```bash
pytest -q
```

The tests check raw loading, expected columns, category normalization, duplicate handling, invalid target/range detection, preservation of invalid rows for validation, processed-data writing, database creation, and execution of every analytical SQL statement.

## Main Task 1 modules

```text
src/data/load_data.py       # project-relative loading + saving
src/data/clean_data.py      # leakage-safe structural cleaning
src/data/validate_data.py   # schema/domain/range/quality validation
scripts/run_data_pipeline.py
scripts/build_database.py
scripts/run_sql_queries.py
sql/schema.sql
sql/queries.sql
notebooks/01_data_pipeline.ipynb
tests/test_data.py
```

## Team handoff

Person 2 (EDA + feature engineering) and Person 3 (classical ML) should start from:

```text
data/processed/processed_data.csv
```

or load it with:

```python
from src.data.load_data import load_processed_data

df = load_processed_data()
```

Before modeling `Machine failure`, exclude identifier columns (`UDI`, `Product ID`) unless there is a justified experiment, and exclude all five failure-mode target columns from predictors to prevent leakage. Train/test splitting, categorical encoding, scaling, class weighting/resampling, and feature engineering should be fit/designed in their own downstream workflows.
