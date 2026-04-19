# Analytics Datathon 2026

Funny but true: the dashboard lives in `dashboard/`, but it is definitely not a
dashboard. It is a local CSV modeling workbench wearing a dashboard costume so
Dash does not get suspicious.

## Abstract

After a comprehensive EDA phase, we ran into a very practical problem: useful
humanitarian data exists, but understanding whether a dataset is worth deeper
analysis can be time intensive. The files often arrive in heterogeneous formats,
with different column names, encodings, category structures, list fields, and
numeric conventions. That makes even a first modeling pass slower than it should
be.

Our idea is to turn that first pass into a fast decision assistant. Instead of
building a fixed dashboard for one schema, the app lets a user upload any CSV,
choose a target, select treatment variables, configure encodings, and train an
exploratory model in a few clicks. The goal is not to produce a final causal
answer. The goal is to quickly judge whether a dataset contains enough signal to
justify deeper work on predicting project impact and understanding which project
features deserve closer attention.

For more detail on the reasoning, assumptions, and approach, see the technical
report: [TECHNICAL REPORT_Not a dashboard.pdf](TECHNICAL%20REPORT_Not%20a%20dashboard.pdf).
That report is the more complete narrative companion to the code and dashboard.

## What Is In This Repo

This repository currently has two main pieces:

- `dashboard/`: an interactive Plotly Dash app for uploading a CSV, choosing a
  numeric target, configuring feature encodings, training a model, reviewing
  diagnostics, and running single-row predictions.
- `scripts/generate_initial_data_quality_report.py`: a standard-library-only
  script that generated the supporting initial data quality appendix under
  `initial_data_quality_appendix/`.
- `TECHNICAL REPORT_Not a dashboard.pdf`: the main project report with more
  detail on the approach, design choices, and interpretation.
- `initial_data_quality_appendix/`: supporting generated data-quality appendix
  files. This is not the main report; use the technical report PDF for the full
  narrative.

The tracked final CSV snapshots are:

- `Final/cerf_ufe_cirv_match_2017_2024.csv`
- `Final/cbpf_projects_allocations.csv`

These files are not hard-coded into the current dashboard UI. To use them in the
app, upload either CSV through the dashboard upload control.

## Dependencies

Dashboard runtime dependencies are listed in both `requirements.txt` and
`pyproject.toml`:

- `dash`
- `numpy`
- `pandas`
- `plotly`
- `scikit-learn`

The initial data quality report generator uses only the Python standard library.
No extra packages are required for `scripts/generate_initial_data_quality_report.py`.

## Run The Dashboard

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 dashboard/app.py
```

Then open:

```text
http://127.0.0.1:8050/
```

The app is local and process-memory based. Uploaded datasets and trained models
live only in the running Dash process.

## Dashboard Workflow

1. Upload a CSV file.
2. Choose a target column. The app tries to preselect an outcome-like numeric
   column, with special preference for CIRV/CIVR-style names.
3. Select one or more treatment variables.
4. Pick an encoding for each selected treatment variable:
   - Numeric scaled
   - Numeric raw
   - One-hot
   - Ordinal
   - Multi-hot literal list
   - Multi-hot delimiter
5. Choose a model:
   - Random Forest, the default
   - Bayesian Ridge, the linear baseline
6. Train the model.
7. Review diagnostics and warnings.
8. Enter feature values for inference and estimate the target with an uncertainty
   interval.

## Model Notes

The dashboard is intended for exploratory modeling, not causal claims. It uses a
random train/test holdout split and reports R2, MAE, mean predictive standard
deviation, target distribution, feature-to-target correlation, pairwise feature
correlation, holdout predicted-vs-actual, and holdout residual plots.

Prediction intervals are model-derived:

- Random Forest: 5th to 95th percentile spread across tree predictions.
- Bayesian Ridge: prediction plus/minus `1.64 * std`, an approximate 90% model
  interval.

## Regenerate The Initial Data Quality Appendix

The report script expects source CSVs under `initial_data/`. If those files are
present locally, run:

```bash
python3 scripts/generate_initial_data_quality_report.py
```

It writes supporting Markdown and CSV appendix tables under
`initial_data_quality_appendix/`.
