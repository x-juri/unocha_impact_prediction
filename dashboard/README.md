# CERF and CBPF CIRV Dashboard

Run the app from the project root:

```bash
python3 -m pip install -r requirements.txt
python3 dashboard/app.py
```

The dashboard reads two local CSV snapshots:

- `Final/CERF_UFE_2017_24_CIRV_Match.csv`
- `Final/Data_ CERF Donor Contributions and Allocations - allocations - CBPFs Projects.csv`

The CBPF file names the target column `CIVR - Inc`; the app treats that as the
same target concept and labels it as `CIRV - Inc` in the UI.

## Approach

Both models are trained at app startup with scikit-learn `BayesianRidge`. The
point estimate is the model mean prediction for `CIRV - Inc`; the interval shown
in the app is an approximate 90% Bayesian Ridge predictive interval. These are
model uncertainty intervals, not causal impact claims.

CERF model features:

- `totalAmountApproved`, parsed as numeric USD and scaled.
- `emergencyTypeName`, `countryName`, and `projectsectors`, one-hot encoded with
  one reference level dropped per field.

CBPF model features:

- `AllocationSourceName`, one-hot encoded with `Reserve` dropped as the binary
  reference category to avoid perfect collinearity.
- `OrganizationType`, one-hot encoded with `International NGO` dropped as the
  reference category.
- `ProjectDuration (Months)`, `Budget`, `Total People`, and `CIRV - Prev`,
  parsed as numeric fields and scaled.
- `Projectsector`, parsed with `ast.literal_eval` as a list and encoded as
  multi-hot indicators. This preserves comma-containing sector names such as
  `Water, Sanitation and Hygiene (WASH)`.

Coefficient notes:

- Numeric coefficients are on standardized feature scale.
- Categorical coefficients are relative to their dropped reference category.
- CBPF project sector coefficients are multi-hot indicator effects.
- Coefficients are associative model terms and should not be read causally.

## Budget Sensecheck

The CBPF `Budget` column is read by pandas as `float64`, and observed raw values
use `.` as the decimal separator. The app strips currency/space characters and
commas before numeric conversion, which introduced zero missing values in the
current file.

| Check | Value |
| --- | ---: |
| Rows | 11,206 |
| Raw dtype | float64 |
| Raw values containing comma | 0 |
| Parsed missing values | 0 |
| Minimum parsed budget | 22,000.00 |
| Median parsed budget | 493,559.17 |
| Mean parsed budget | 696,359.14 |
| Maximum parsed budget | 29,499,999.99 |

Sample raw values: `102640.61`, `2125691.55`, `463266.3`, `550889.55`,
`719215.74`, `384761.0`, `248685.58`, `1185709.59`.

## Model Diagnostics

| Model | Train rows | Test rows | Test R2 | Test MAE | Mean test std |
| --- | ---: | ---: | ---: | ---: | ---: |
| CERF Bayesian Ridge | 768 | 256 | 0.2109 | 0.0960 | 0.1342 |
| CBPF Bayesian Ridge | 8,404 | 2,802 | 0.0268 | 0.0950 | 0.1273 |

## Correlation and Holdout Diagnostics Views

Each EDA section now includes correlation diagnostics tied to the active filters:

- Feature-to-target correlation bar chart (Pearson or Spearman).
- Pairwise feature correlation heatmap based on top-N correlated features.
- User-selectable top-N feature cap (default `Top 5`).

Each EDA section also includes holdout evaluation plots built from the model test
split:

- Predicted vs actual plot with a model-based 90% uncertainty ribbon.
- Residuals vs predictions plot with a model-based 90% uncertainty ribbon.

The ribbon uses the same Bayesian interval semantics shown elsewhere in the app
(`prediction ± 1.64 * std`), aggregated over quantile bins for readability.

## Top Regression Coefficients

Top coefficients are sorted by absolute coefficient size.

### CERF

Reference categories: `emergencyTypeName=Cholera`, `countryName=Afghanistan`,
and `projectsectors=Agriculture`.

| Feature | Coefficient |
| --- | ---: |
| countryName_Republic of Congo | -0.2312 |
| countryName_Lebanon | 0.1980 |
| countryName_Jordan | 0.1717 |
| countryName_Myanmar | 0.1287 |
| countryName_Ukraine | -0.1065 |
| countryName_Venezuela Regional Refugee and Migration Crisis | 0.1059 |
| countryName_Democratic People's Republic of Korea | -0.1000 |
| projectsectors_Protection; Shelter and Non-Food Items; Protection | -0.0953 |
| projectsectors_Common Services; Food Assistance | -0.0953 |
| projectsectors_Nutrition; Water; Sanitation and Hygiene; Education; Protection; Shelter and Non-Food Items | -0.0953 |
| countryName_Philippines | 0.0923 |
| countryName_Haiti | 0.0918 |
| countryName_Rwanda | -0.0894 |
| projectsectors_Nutrition; Water; Sanitation and Hygiene; Health | 0.0874 |
| countryName_Central African Republic | -0.0792 |
| countryName_Republic of the Sudan | -0.0663 |
| countryName_Venezuela | 0.0662 |
| emergencyTypeName_Economic Disruption | -0.0653 |
| projectsectors_Multi-Sector | -0.0629 |
| countryName_Nigeria | -0.0620 |

### CBPF

Reference categories: `AllocationSourceName=Reserve` and
`OrganizationType=International NGO`.

| Feature | Coefficient |
| --- | ---: |
| Projectsector: Other | 0.0285 |
| cirv_prev | -0.0181 |
| Projectsector: Camp Coordination | -0.0160 |
| Projectsector: Protection | 0.0115 |
| project_duration_months | 0.0093 |
| Projectsector: Emergency Shelter and Non-Food Items | 0.0076 |
| OrganizationType_Others | -0.0067 |
| Projectsector: Nutrition | 0.0053 |
| Projectsector: Logistics | 0.0046 |
| Projectsector: Education | 0.0042 |
| Projectsector: Health | -0.0042 |
| Projectsector: Early Recovery | 0.0039 |
| Projectsector: Water, Sanitation and Hygiene (WASH) | 0.0039 |
| budget_numeric | 0.0037 |
| AllocationSourceName_Standard | -0.0037 |
| OrganizationType_UN Agency | -0.0034 |
| total_people_numeric | -0.0018 |
| OrganizationType_National NGO | 0.0017 |
| Projectsector: Food Security | -0.0013 |
| Projectsector: Emergency Telecommunications | -0.0000 |

## Smoke Test

To test preprocessing, model training, budget parsing, project-sector parsing,
and uncertainty-producing predictions without starting Dash:

```bash
python3 -B dashboard/smoke_test_pipeline.py
```
