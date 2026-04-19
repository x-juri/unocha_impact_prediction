# CSV Modeling Workbench

This folder contains the Dash app. It looks like a dashboard, acts like a
dashboard, and yet its real job is to let you upload arbitrary CSVs and quickly
train exploratory models.

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 dashboard/app.py
```

Open `http://127.0.0.1:8050/`.

## Current App Behavior

The app is no longer a fixed CERF/CBPF-only estimator. It is a general CSV
modeling workflow:

1. Upload one CSV file.
2. Select a numeric target column.
3. Select treatment variables.
4. Configure each treatment encoding.
5. Train either Random Forest or Bayesian Ridge.
6. Inspect diagnostics and adaptive warning messages.
7. Enter values for a single-row prediction.

Uploaded datasets and trained models are stored in small in-memory caches. They
are not written to disk and are cleared when the Dash process stops.

## Supported Encodings

- `Numeric (scaled)`: numeric parsing, median fill, standard scaling.
- `Numeric (raw)`: numeric parsing and median fill without scaling.
- `One-hot`: categorical encoding with one reference category dropped.
- `Ordinal`: sorted category labels mapped to integer codes.
- `Multi-hot (literal list)`: list-like cells such as `['Health', 'WASH']`.
- `Multi-hot (delimiter)`: string cells split by a selected delimiter.

## Models

The model selector currently supports:

- `Random Forest`: default model. Intervals use the 5th to 95th percentile
  spread across tree predictions.
- `Bayesian Ridge`: linear baseline. Intervals use the model predictive
  standard deviation with `1.64 * std`.

Both models use the same encoded feature matrix and a random holdout split for
diagnostics.

## Diagnostics

After training, the app renders:

- Test R2, test MAE, mean test predictive standard deviation, and row count.
- Target distribution.
- Top feature-to-target correlations.
- Pairwise correlation heatmap for the selected top features.
- Holdout predicted-vs-actual plot.
- Holdout residual plot.
- Warning messages for weak fit, overfitting risk, small holdouts, many encoded
  features, and stale model selections.

## Final CSV Snapshots

The two tracked final CSV snapshots have readable names:

- `Final/cerf_ufe_cirv_match_2017_2024.csv`
- `Final/cbpf_projects_allocations.csv`

To model either file, upload it through the app. The current dashboard does not
load these files automatically at startup.

## Smoke Test

Run the lightweight pipeline smoke test without starting Dash:

```bash
python3 -B dashboard/smoke_test_pipeline.py
```
