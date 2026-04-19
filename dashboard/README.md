# On-the-go CSV Modeling Dashboard

Run the app from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 dashboard/app.py
```

## MVP Workflow

1. Upload a CSV file.
2. Select one target column and one or more treatment variables.
3. Configure encoding per treatment variable:
   - Numeric (scaled)
   - Numeric (raw)
   - One-hot
   - Ordinal
   - Multi-hot (literal list)
   - Multi-hot (delimiter)
4. Train the Bayesian Ridge model.
5. Inspect diagnostics and run inference with model uncertainty intervals.

## Diagnostics Included

- Metrics cards (Test R2, Test MAE, mean test std, rows)
- Target distribution
- Feature-to-target correlation (Pearson/Spearman with top-N)
- Pairwise feature correlation heatmap
- Holdout predicted vs actual with model-based ribbon
- Holdout residuals with model-based ribbon

The ribbon semantics are `prediction ± 1.64 * std` (90% model interval).

## Encoding Notes

- **One-hot** drops one reference category.
- **Ordinal** assigns sorted integer codes to categories.
- **Multi-hot literal** expects list-like values, e.g. `['Health', 'WASH']`.
- **Multi-hot delimiter** splits string values by a selected delimiter.

## MVP Compatibility Goal

The app is designed to work with the two datasets previously used in this repo
when uploaded as CSV:

- `Final/CERF_UFE_2017_24_CIRV_Match.csv`
- `Final/Data_ CERF Donor Contributions and Allocations - allocations - CBPFs Projects.csv`
