# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source .venv/bin/activate   # Python 3.9.6 virtual environment
```

Key libraries already installed: `pandas`, `numpy`, `openpyxl`, `pdfplumber`, `jupyter`.

## Common Commands

```bash
# Run the main merge pipeline
python scripts/merge_cerf_cirv_2022_2025.py

# Launch notebook for ETL exploration
jupyter lab notebooks/01-data-set-generation.ipynb

# Validation scripts (run after pipeline)
python scratch/final_math_check.py    # Validates CSV totals against PDF allocation amounts
python scratch/compare_totals.py      # Reconciles funding totals by year/round
python scratch/verify_merge.py        # Checks join coverage (matched/unmatched rows)

# GDELT crisis media coverage pipeline (run in order)
python scripts/fetch_gdelt_coverage.py   # Downloads sampled GDELT data → initial_data/gdelt_crisis_coverage.csv
python scripts/plot_gdelt_cerf.py        # Scatter plot of CNI vs CERF funding → reports/gdelt_cerf_scatter.png
```

There is no test framework — validation is done by comparison against official PDF allocation documents.

## Architecture

The project is a data ETL pipeline merging two humanitarian datasets for analysis:

1. **HDX CERF Allocations** (`initial_data/hdx_cerf_allocations.csv`) — 9,849 rows, all CERF allocations 2006–2025, 21+ columns (agency, country, sector, project, funding amounts). This is the authoritative base table.
2. **CIRV Indicator Tables** (`initial_data/cirv/`) — Round-wise crisis vulnerability scores (Excl. Inform, Incl. Inform, Adjusted) per country, stored as Excel (2022–2024) and CSV (2025-II).

**Pipeline flow:**
```
HDX base table → filter UFE 2022-2025 → assign Round (I/II) by signature month
    → load CIRV Excel/CSV → left join on (ISO3, year, round)
    → final_merged_cerf_cirv_2022_2025.csv (2,449 rows)
```

**Key design decisions:**
- **Temporal binning:** signature date month ≤ 6 → Round I, > 6 → Round II. UFE round assignments verified against official PDF allocation decision documents in `initial_data/allocations/`.
- **Year extraction:** Regex on `projectCode` field to extract year (handles formats like `22-UF-FAO-...` and `CERF-SDN-25-UF-...`).
- **Outer join:** Preserves HDX rows without CIRV matches (funding with no vulnerability data) and orphaned CIRV rows (vulnerability data without matched funding).
- **Row-level granularity:** All 21+ HDX columns are preserved rather than aggregated, enabling post-hoc grouping by agency, sector, region, etc.

**Output files:**
- `initial_data/unified_cerf_2022_2024.csv` — 399 rows, produced by the notebook
- `initial_data/final_merged_cerf_cirv_2022_2025.csv` — 2,449 rows, produced by the script (primary output)

**Reports:** `reports/initial_data_quality_report.md` is a 156 KB comprehensive data quality report covering missingness, duplicates, cross-dataset join coverage, and a "What We Can And Cannot Say" matrix — consult it before drawing analytical conclusions.

## Data File Conventions

- `.xlsx`, `.parquet`, `.jsonl`, `.h5`, `.zip`, `.tar.gz` files are gitignored — only CSV and PDF source files are tracked.
- `scratch/` contains exploratory one-off scripts; `scripts/` contains the production pipeline.
- CIRV Excel files have variable sheet layouts per year — the merge script handles sheet/column detection dynamically.
