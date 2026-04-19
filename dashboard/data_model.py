from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CERF_DATA_PATH = PROJECT_ROOT / "Final" / "CERF_UFE_2017_24_CIRV_Match.csv"
CBPF_DATA_PATH = (
    PROJECT_ROOT
    / "Final"
    / "Data_ CERF Donor Contributions and Allocations - allocations - CBPFs Projects.csv"
)
DATA_PATH = CERF_DATA_PATH

TARGET_LABEL = "CIRV - Inc"
TARGET_COLUMN = "CIRV - Inc"
TARGET_NUMERIC_COLUMN = "cirv_inc"
CIRV_PREV_COLUMN = "CIRV - Prev"
CIRV_PREV_NUMERIC_COLUMN = "cirv_prev"

AMOUNT_COLUMN = "totalAmountApproved"
AMOUNT_NUMERIC_COLUMN = "totalAmountApproved_numeric"
CERF_CATEGORICAL_FEATURES = ["emergencyTypeName", "countryName", "projectsectors"]
CATEGORICAL_FEATURES = CERF_CATEGORICAL_FEATURES
CERF_MODEL_FEATURES = [AMOUNT_NUMERIC_COLUMN, *CERF_CATEGORICAL_FEATURES]
MODEL_FEATURES = CERF_MODEL_FEATURES
MODEL_BAYESIAN_RIDGE = "bayesian_ridge"
MODEL_RANDOM_FOREST = "random_forest"
DEFAULT_MODEL_KEY = MODEL_RANDOM_FOREST
MODEL_LABELS = {
    MODEL_RANDOM_FOREST: "Random Forest",
    MODEL_BAYESIAN_RIDGE: "Bayesian Ridge",
}
MODEL_OPTIONS = [
    {"label": MODEL_LABELS[MODEL_RANDOM_FOREST], "value": MODEL_RANDOM_FOREST},
    {"label": MODEL_LABELS[MODEL_BAYESIAN_RIDGE], "value": MODEL_BAYESIAN_RIDGE},
]
AVAILABLE_MODEL_KEYS = [option["value"] for option in MODEL_OPTIONS]
CERF_REQUIRED_COLUMNS = [
    AMOUNT_COLUMN,
    TARGET_COLUMN,
    CIRV_PREV_COLUMN,
    "year",
    *CERF_CATEGORICAL_FEATURES,
]

CBPF_TARGET_COLUMN = "CIVR - Inc"
CBPF_ALLOCATION_SOURCE_COLUMN = "AllocationSourceName"
CBPF_ORGANIZATION_TYPE_COLUMN = "OrganizationType"
CBPF_DURATION_COLUMN = "ProjectDuration (Months)"
CBPF_DURATION_NUMERIC_COLUMN = "project_duration_months"
CBPF_BUDGET_COLUMN = "Budget"
CBPF_BUDGET_NUMERIC_COLUMN = "budget_numeric"
CBPF_BUDGET_RAW_COLUMN = "budget_raw"
CBPF_PROJECT_SECTOR_COLUMN = "Projectsector"
CBPF_PROJECT_SECTOR_LIST_COLUMN = "projectsector_list"
CBPF_SECTOR_PREFIX = "sector__"
CBPF_TOTAL_PEOPLE_COLUMN = "Total People"
CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN = "total_people_numeric"
CBPF_YEAR_COLUMN = "AllocationYear"
CBPF_COUNTRY_COLUMN = "Country"
CBPF_CATEGORICAL_FEATURES = [CBPF_ALLOCATION_SOURCE_COLUMN, CBPF_ORGANIZATION_TYPE_COLUMN]
CBPF_NUMERIC_FEATURES = [
    CBPF_DURATION_NUMERIC_COLUMN,
    CBPF_BUDGET_NUMERIC_COLUMN,
    CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
    CIRV_PREV_NUMERIC_COLUMN,
]
CBPF_REQUIRED_COLUMNS = [
    CBPF_COUNTRY_COLUMN,
    CBPF_YEAR_COLUMN,
    CBPF_ALLOCATION_SOURCE_COLUMN,
    CBPF_ORGANIZATION_TYPE_COLUMN,
    CBPF_DURATION_COLUMN,
    CBPF_BUDGET_COLUMN,
    CBPF_PROJECT_SECTOR_COLUMN,
    CBPF_TOTAL_PEOPLE_COLUMN,
    CIRV_PREV_COLUMN,
    CBPF_TARGET_COLUMN,
]


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]
    target_column: str
    holdout_diagnostics: pd.DataFrame
    interval_z: float = 1.64


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    lower: float
    upper: float
    std: float


def _dense_onehot_encoder(drop: str | None = None) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False)


def _parse_decimal_comma(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_general_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[$\s]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    direct = pd.to_numeric(series, errors="coerce")
    general = _parse_general_number(series)
    decimal_comma = _parse_decimal_comma(series)
    candidates = {
        "direct": direct,
        "general": general,
        "decimal_comma": decimal_comma,
    }
    scores = {name: int(values.notna().sum()) for name, values in candidates.items()}
    best_name = max(scores, key=scores.get)

    comma_mask = text.str.contains(",", regex=False, na=False)
    dot_mask = text.str.contains(".", regex=False, na=False)
    comma_no_dot = comma_mask & ~dot_mask
    if comma_no_dot.any():
        right_lengths = text[comma_no_dot].str.rsplit(",", n=1).str[-1].str.len()
        decimal_like = int((right_lengths <= 2).sum())
        thousands_like = int((right_lengths == 3).sum())
        if decimal_like > thousands_like and scores["decimal_comma"] >= max(scores["general"] - 2, 0):
            best_name = "decimal_comma"
        elif thousands_like > decimal_like and scores["general"] >= max(scores["decimal_comma"] - 2, 0):
            best_name = "general"

    return candidates[best_name]


def _normalise_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.replace("", pd.NA).fillna("Unknown").astype(str)


def parse_project_sector_list(value: object) -> list[str]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, tuple):
        parsed = list(value)
    elif pd.isna(value):
        return []
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return [stripped]
    else:
        return [str(value)]

    if isinstance(parsed, tuple):
        parsed = list(parsed)
    if not isinstance(parsed, list):
        return [str(parsed).strip()] if str(parsed).strip() else []

    sectors = []
    for item in parsed:
        label = str(item).strip()
        if label:
            sectors.append(label)
    return sectors


def load_and_clean_cerf_data(path: Path | str = CERF_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = sorted(set(CERF_REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"CERF CSV is missing required columns: {missing_columns}")

    cleaned = df.copy()
    cleaned[TARGET_NUMERIC_COLUMN] = _parse_decimal_comma(cleaned[TARGET_COLUMN])
    cleaned[CIRV_PREV_NUMERIC_COLUMN] = _parse_decimal_comma(cleaned[CIRV_PREV_COLUMN])
    cleaned[AMOUNT_NUMERIC_COLUMN] = _parse_general_number(cleaned[AMOUNT_COLUMN])
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")

    for column in CERF_CATEGORICAL_FEATURES:
        cleaned[column] = _normalise_category(cleaned[column])

    valid = cleaned[TARGET_NUMERIC_COLUMN].notna() & cleaned[AMOUNT_NUMERIC_COLUMN].notna()
    valid &= cleaned["year"].notna()
    return cleaned.loc[valid].copy()


def load_and_clean_data(path: Path | str = CERF_DATA_PATH) -> pd.DataFrame:
    return load_and_clean_cerf_data(path)


def load_and_clean_cbpf_projects(path: Path | str = CBPF_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = sorted(set(CBPF_REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"CBPF projects CSV is missing required columns: {missing_columns}")

    cleaned = df.copy()
    cleaned[CBPF_BUDGET_RAW_COLUMN] = cleaned[CBPF_BUDGET_COLUMN].astype("string")
    cleaned[TARGET_NUMERIC_COLUMN] = _parse_decimal_comma(cleaned[CBPF_TARGET_COLUMN])
    cleaned[CIRV_PREV_NUMERIC_COLUMN] = _parse_decimal_comma(cleaned[CIRV_PREV_COLUMN])
    cleaned[CBPF_BUDGET_NUMERIC_COLUMN] = _parse_general_number(cleaned[CBPF_BUDGET_COLUMN])
    cleaned[CBPF_DURATION_NUMERIC_COLUMN] = pd.to_numeric(
        cleaned[CBPF_DURATION_COLUMN],
        errors="coerce",
    )
    cleaned[CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN] = _parse_general_number(
        cleaned[CBPF_TOTAL_PEOPLE_COLUMN]
    )
    cleaned[CBPF_YEAR_COLUMN] = pd.to_numeric(cleaned[CBPF_YEAR_COLUMN], errors="coerce")

    for column in [CBPF_COUNTRY_COLUMN, *CBPF_CATEGORICAL_FEATURES]:
        cleaned[column] = _normalise_category(cleaned[column])

    cleaned[CBPF_PROJECT_SECTOR_LIST_COLUMN] = cleaned[CBPF_PROJECT_SECTOR_COLUMN].apply(
        parse_project_sector_list
    )
    sector_names = cbpf_sector_names(cleaned)
    for sector in sector_names:
        column = sector_feature_column(sector)
        cleaned[column] = cleaned[CBPF_PROJECT_SECTOR_LIST_COLUMN].apply(
            lambda values, sector=sector: int(sector in values)
        )

    required_numeric = [
        TARGET_NUMERIC_COLUMN,
        CIRV_PREV_NUMERIC_COLUMN,
        CBPF_BUDGET_NUMERIC_COLUMN,
        CBPF_DURATION_NUMERIC_COLUMN,
        CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
        CBPF_YEAR_COLUMN,
    ]
    valid = cleaned[required_numeric].notna().all(axis=1)
    return cleaned.loc[valid].copy()


def cbpf_sector_names(df: pd.DataFrame) -> list[str]:
    sectors = set()
    for values in df[CBPF_PROJECT_SECTOR_LIST_COLUMN]:
        sectors.update(values)
    return sorted(sectors)


def sector_feature_column(sector: str) -> str:
    return f"{CBPF_SECTOR_PREFIX}{sector}"


def cbpf_sector_feature_columns(df: pd.DataFrame) -> list[str]:
    return [sector_feature_column(sector) for sector in cbpf_sector_names(df)]


def build_cerf_regression_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _dense_onehot_encoder(drop="first")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, [AMOUNT_NUMERIC_COLUMN]),
            ("categorical", categorical_pipeline, CERF_CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", BayesianRidge()),
        ]
    )


def build_regression_pipeline() -> Pipeline:
    return build_cerf_regression_pipeline()


def build_cbpf_regression_pipeline(sector_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    allocation_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _dense_onehot_encoder(drop="first")),
        ]
    )
    organization_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _dense_onehot_encoder(drop="first")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, CBPF_NUMERIC_FEATURES),
            ("allocation_source", allocation_pipeline, [CBPF_ALLOCATION_SOURCE_COLUMN]),
            ("organization_type", organization_pipeline, [CBPF_ORGANIZATION_TYPE_COLUMN]),
            ("project_sector", "passthrough", sector_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", BayesianRidge()),
        ]
    )


def _train_model_bundle(
    df: pd.DataFrame,
    feature_columns: list[str],
    pipeline: Pipeline,
    target_column: str = TARGET_NUMERIC_COLUMN,
    random_state: int = 42,
    interval_z: float = 1.64,
) -> ModelBundle:
    model_data = df.dropna(subset=[target_column, *feature_columns]).copy()
    if model_data.empty:
        raise ValueError("No complete rows are available for model training.")

    x = model_data[feature_columns]
    y = model_data[target_column]
    holdout_diagnostics = model_data.iloc[0:0].copy()
    holdout_diagnostics["source_index"] = pd.Series(dtype="int64")
    holdout_diagnostics["actual"] = pd.Series(dtype="float64")
    holdout_diagnostics["prediction"] = pd.Series(dtype="float64")
    holdout_diagnostics["std"] = pd.Series(dtype="float64")
    holdout_diagnostics["lower"] = pd.Series(dtype="float64")
    holdout_diagnostics["upper"] = pd.Series(dtype="float64")
    holdout_diagnostics["residual"] = pd.Series(dtype="float64")

    if len(model_data) >= 8:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=random_state,
        )
        evaluation_pipeline = clone(pipeline)
        evaluation_pipeline.fit(x_train, y_train)
        train_predictions = _safe_predict(evaluation_pipeline, x_train)
        test_predictions = _safe_predict(evaluation_pipeline, x_test)
        train_uncertainty = predict_with_uncertainty_from_frame(evaluation_pipeline, x_train)
        test_uncertainty = predict_with_uncertainty_from_frame(evaluation_pipeline, x_test)

        holdout_diagnostics = model_data.loc[x_test.index].copy()
        holdout_diagnostics["source_index"] = holdout_diagnostics.index
        holdout_diagnostics["actual"] = y_test.to_numpy()
        holdout_diagnostics["prediction"] = test_predictions
        holdout_diagnostics["std"] = test_uncertainty["std"].to_numpy()
        holdout_diagnostics["lower"] = (
            holdout_diagnostics["prediction"] - interval_z * holdout_diagnostics["std"]
        )
        holdout_diagnostics["upper"] = (
            holdout_diagnostics["prediction"] + interval_z * holdout_diagnostics["std"]
        )
        holdout_diagnostics["residual"] = (
            holdout_diagnostics["actual"] - holdout_diagnostics["prediction"]
        )
        metrics = {
            "train_r2": float(r2_score(y_train, train_predictions)),
            "test_r2": float(r2_score(y_test, test_predictions)),
            "train_mae": float(mean_absolute_error(y_train, train_predictions)),
            "test_mae": float(mean_absolute_error(y_test, test_predictions)),
            "train_mean_std": float(train_uncertainty["std"].mean()),
            "test_mean_std": float(test_uncertainty["std"].mean()),
            "train_rows": float(len(x_train)),
            "test_rows": float(len(x_test)),
        }
    else:
        metrics = {
            "train_r2": float("nan"),
            "test_r2": float("nan"),
            "train_mae": float("nan"),
            "test_mae": float("nan"),
            "train_mean_std": float("nan"),
            "test_mean_std": float("nan"),
            "train_rows": float(len(model_data)),
            "test_rows": 0.0,
        }

    final_pipeline = clone(pipeline)
    final_pipeline.fit(x, y)
    return ModelBundle(
        pipeline=final_pipeline,
        metrics=metrics,
        feature_columns=feature_columns,
        target_column=target_column,
        holdout_diagnostics=holdout_diagnostics,
        interval_z=interval_z,
    )


def train_cerf_model(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    return _train_model_bundle(
        df=df,
        feature_columns=CERF_MODEL_FEATURES,
        pipeline=build_cerf_regression_pipeline(),
        random_state=random_state,
    )


def train_model(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    return train_cerf_model(df, random_state=random_state)


def train_cbpf_model(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    sector_columns = cbpf_sector_feature_columns(df)
    return _train_model_bundle(
        df=df,
        feature_columns=[*CBPF_NUMERIC_FEATURES, *CBPF_CATEGORICAL_FEATURES, *sector_columns],
        pipeline=build_cbpf_regression_pipeline(sector_columns),
        random_state=random_state,
    )


def predict_with_uncertainty_from_frame(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["regressor"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
            category=UserWarning,
        )
        transformed = preprocessor.transform(frame)
        predictions, std = regressor.predict(transformed, return_std=True)
    return pd.DataFrame({"prediction": predictions, "std": std})


def _safe_predict(pipeline: Pipeline, frame: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
            category=UserWarning,
        )
        return pipeline.predict(frame)


def predict_with_uncertainty(bundle: ModelBundle, frame: pd.DataFrame) -> PredictionResult:
    result = predict_with_uncertainty_from_frame(bundle.pipeline, frame).iloc[0]
    prediction = float(result["prediction"])
    std = float(result["std"])
    return PredictionResult(
        prediction=prediction,
        lower=prediction - bundle.interval_z * std,
        upper=prediction + bundle.interval_z * std,
        std=std,
    )


def make_cerf_prediction_frame(
    total_amount_approved: float,
    emergency_type: str,
    country: str,
    project_sectors: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                AMOUNT_NUMERIC_COLUMN: float(total_amount_approved),
                "emergencyTypeName": emergency_type,
                "countryName": country,
                "projectsectors": project_sectors,
            }
        ]
    )


def predict_impact(
    model: Pipeline,
    total_amount_approved: float,
    emergency_type: str,
    country: str,
    project_sectors: str,
) -> float:
    prediction_input = make_cerf_prediction_frame(
        total_amount_approved,
        emergency_type,
        country,
        project_sectors,
    )
    return float(model.predict(prediction_input)[0])


def make_cbpf_prediction_frame(
    sector_columns: list[str],
    allocation_source: str,
    organization_type: str,
    project_duration_months: float,
    budget: float,
    total_people: float,
    cirv_prev: float,
    project_sectors: list[str],
) -> pd.DataFrame:
    row = {
        CBPF_ALLOCATION_SOURCE_COLUMN: allocation_source,
        CBPF_ORGANIZATION_TYPE_COLUMN: organization_type,
        CBPF_DURATION_NUMERIC_COLUMN: float(project_duration_months),
        CBPF_BUDGET_NUMERIC_COLUMN: float(budget),
        CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN: float(total_people),
        CIRV_PREV_NUMERIC_COLUMN: float(cirv_prev),
    }
    selected = set(project_sectors)
    for column in sector_columns:
        sector = column.removeprefix(CBPF_SECTOR_PREFIX)
        row[column] = int(sector in selected)
    return pd.DataFrame([row])


def feature_coefficients(bundle: ModelBundle) -> pd.DataFrame:
    feature_names = bundle.pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = bundle.pipeline.named_steps["regressor"].coef_
    return (
        pd.DataFrame(
            {
                "feature": [_clean_feature_name(name) for name in feature_names],
                "coefficient": coefficients,
            }
        )
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def _clean_feature_name(name: str) -> str:
    clean = name
    for prefix in [
        "numeric__",
        "categorical__",
        "allocation_source__",
        "organization_type__",
        "project_sector__",
    ]:
        clean = clean.replace(prefix, "")
    return clean.replace(CBPF_SECTOR_PREFIX, "Projectsector: ")


def dropdown_options(values: Iterable[object]) -> list[dict[str, object]]:
    clean_values = sorted({str(value) for value in values if pd.notna(value)})
    return [{"label": value, "value": value} for value in clean_values]


def year_options(df: pd.DataFrame, column: str = "year") -> list[dict[str, object]]:
    years = sorted(int(year) for year in df[column].dropna().unique())
    return [{"label": str(year), "value": year} for year in years]


def budget_sensecheck(df: pd.DataFrame) -> dict[str, object]:
    raw = df[CBPF_BUDGET_RAW_COLUMN].astype("string")
    parsed = df[CBPF_BUDGET_NUMERIC_COLUMN]
    return {
        "raw_dtype": str(df[CBPF_BUDGET_COLUMN].dtype),
        "rows": int(len(df)),
        "raw_values_with_comma": int(raw.str.contains(",", regex=False, na=False).sum()),
        "parsed_missing": int(parsed.isna().sum()),
        "min": float(parsed.min()),
        "median": float(parsed.median()),
        "mean": float(parsed.mean()),
        "max": float(parsed.max()),
        "examples": raw.dropna().head(8).tolist(),
    }


@dataclass(frozen=True)
class FeatureEncodingSpec:
    column: str
    encoding: str
    delimiter: str | None = None


@dataclass(frozen=True)
class FittedFeatureSpec:
    column: str
    encoding: str
    feature_names: list[str]
    fill_value: float | None = None
    mean: float | None = None
    std: float | None = None
    categories: list[str] | None = None
    token_vocabulary: list[str] | None = None
    delimiter: str | None = None
    default_value: object | None = None


@dataclass(frozen=True)
class DynamicModelBundle:
    regressor: BayesianRidge | RandomForestRegressor
    target_column: str
    feature_specs: list[FittedFeatureSpec]
    encoded_feature_names: list[str]
    metrics: dict[str, float]
    holdout_diagnostics: pd.DataFrame
    training_encoded: pd.DataFrame
    training_target: pd.Series
    model_key: str = MODEL_BAYESIAN_RIDGE
    model_label: str = MODEL_LABELS[MODEL_BAYESIAN_RIDGE]
    interval_z: float = 1.64


def _regressor_for_model(model_key: str, random_state: int = 42):
    if model_key == MODEL_BAYESIAN_RIDGE:
        return BayesianRidge()
    if model_key == MODEL_RANDOM_FOREST:
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model key: {model_key}")


def _predict_with_uncertainty_matrix(
    regressor,
    frame: pd.DataFrame,
    interval_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(regressor, "estimators_"):
        matrix = frame.to_numpy(copy=False)
        tree_predictions = np.vstack([estimator.predict(matrix) for estimator in regressor.estimators_])
        predictions = regressor.predict(frame)
        std = tree_predictions.std(axis=0, ddof=1)
        lower = np.percentile(tree_predictions, 5, axis=0)
        upper = np.percentile(tree_predictions, 95, axis=0)
        return predictions, std, lower, upper

    predictions, std = regressor.predict(frame, return_std=True)
    lower = predictions - interval_z * std
    upper = predictions + interval_z * std
    return predictions, std, lower, upper


def detect_column_role(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"

    numeric_ratio = _coerce_numeric(non_null).notna().mean()
    if numeric_ratio >= 0.9:
        return "numeric"

    string_values = non_null.astype("string").str.strip()
    if string_values.empty:
        return "categorical"

    literal_hits = 0
    inspected = 0
    semicolon_hits = 0
    comma_hits = 0
    for raw in string_values.head(500):
        if not raw:
            continue
        inspected += 1
        if ";" in raw:
            semicolon_hits += 1
        if "," in raw:
            comma_hits += 1
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            literal_hits += 1

    if inspected == 0:
        return "categorical"
    if literal_hits / inspected >= 0.5:
        return "multi_hot_literal"
    if max(semicolon_hits, comma_hits) / inspected >= 0.5:
        return "multi_hot_delimited"
    return "categorical"


def default_encoding_for_series(series: pd.Series) -> FeatureEncodingSpec:
    role = detect_column_role(series)
    if role == "numeric":
        return FeatureEncodingSpec(column=str(series.name), encoding="numeric_scaled")
    if role == "multi_hot_literal":
        return FeatureEncodingSpec(column=str(series.name), encoding="multi_hot_literal")
    if role == "multi_hot_delimited":
        delimiter = ";"
        sample = series.dropna().astype("string").head(200)
        semicolon_hits = sample.str.contains(";", regex=False).sum()
        comma_hits = sample.str.contains(",", regex=False).sum()
        if comma_hits > semicolon_hits:
            delimiter = ","
        return FeatureEncodingSpec(column=str(series.name), encoding="multi_hot_delimited", delimiter=delimiter)
    return FeatureEncodingSpec(column=str(series.name), encoding="one_hot")


def infer_column_roles(df: pd.DataFrame) -> dict[str, str]:
    return {column: detect_column_role(df[column]) for column in df.columns}


def validate_training_setup(
    df: pd.DataFrame,
    target_column: str | None,
    treatment_columns: list[str] | None,
) -> list[str]:
    errors: list[str] = []
    if target_column is None or not target_column:
        errors.append("Select a target column.")
    elif target_column not in df.columns:
        errors.append(f"Target column '{target_column}' is not present in the dataset.")

    treatments = treatment_columns or []
    if not treatments:
        errors.append("Select at least one treatment variable.")
    for column in treatments:
        if column not in df.columns:
            errors.append(f"Treatment column '{column}' is not present in the dataset.")
    if target_column and target_column in treatments:
        errors.append("Target column cannot also be a treatment variable.")
    return errors


def _normalise_string(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.replace("", pd.NA).fillna("Unknown").astype(str)


def _parse_multi_hot_literal(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        parsed = list(value)
    elif pd.isna(value):
        return []
    else:
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    if not isinstance(parsed, list):
        parsed = [parsed]
    tokens: list[str] = []
    for item in parsed:
        token = str(item).strip()
        if token:
            tokens.append(token)
    return tokens


def _parse_multi_hot_delimited(value: object, delimiter: str | None) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(token).strip() for token in value if str(token).strip()]
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    delimiter = delimiter or ";"
    return [token.strip() for token in raw.split(delimiter) if token.strip()]


def _fit_feature_spec(series: pd.Series, spec: FeatureEncodingSpec) -> tuple[FittedFeatureSpec, pd.DataFrame]:
    column = spec.column
    encoding = spec.encoding
    index = series.index

    if encoding in {"numeric_scaled", "numeric_raw"}:
        numeric = _coerce_numeric(series)
        if numeric.notna().any():
            fill = float(numeric.median())
        else:
            fill = 0.0
        filled = numeric.fillna(fill).astype(float)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if std <= 0:
            std = 1.0
        transformed = (filled - mean) / std if encoding == "numeric_scaled" else filled
        frame = pd.DataFrame({column: transformed}, index=index)
        fitted = FittedFeatureSpec(
            column=column,
            encoding=encoding,
            feature_names=[column],
            fill_value=fill,
            mean=mean,
            std=std,
            default_value=fill,
        )
        return fitted, frame

    if encoding == "one_hot":
        values = _normalise_string(series)
        categories = sorted(values.unique().tolist())
        reference = categories[0] if categories else "Unknown"
        active_categories = categories[1:] if len(categories) > 1 else []
        encoded_map = {f"{column}={category}": (values == category).astype(float) for category in active_categories}
        frame = pd.DataFrame(encoded_map, index=index)
        fitted = FittedFeatureSpec(
            column=column,
            encoding=encoding,
            feature_names=list(frame.columns),
            categories=categories,
            default_value=reference,
        )
        return fitted, frame

    if encoding == "ordinal":
        values = _normalise_string(series)
        categories = sorted(values.unique().tolist())
        mapping = {category: float(idx) for idx, category in enumerate(categories)}
        encoded = values.map(mapping).fillna(-1.0).astype(float)
        frame = pd.DataFrame({column: encoded}, index=index)
        default_value = categories[0] if categories else "Unknown"
        fitted = FittedFeatureSpec(
            column=column,
            encoding=encoding,
            feature_names=[column],
            categories=categories,
            default_value=default_value,
        )
        return fitted, frame

    if encoding in {"multi_hot_literal", "multi_hot_delimited"}:
        parser = _parse_multi_hot_literal
        if encoding == "multi_hot_delimited":
            parser = lambda value: _parse_multi_hot_delimited(value, spec.delimiter)
        tokens_per_row = series.apply(parser)
        vocabulary = sorted({token for row_tokens in tokens_per_row for token in row_tokens})
        encoded_map = {
            f"{column} contains {token}": tokens_per_row.apply(lambda values, token=token: float(token in values))
            for token in vocabulary
        }
        frame = pd.DataFrame(encoded_map, index=index)
        fitted = FittedFeatureSpec(
            column=column,
            encoding=encoding,
            feature_names=list(frame.columns),
            token_vocabulary=vocabulary,
            delimiter=spec.delimiter,
            default_value=[],
        )
        return fitted, frame

    raise ValueError(f"Unsupported encoding '{encoding}' for column '{column}'.")


def _transform_with_fitted_specs(df: pd.DataFrame, specs: list[FittedFeatureSpec]) -> pd.DataFrame:
    encoded_parts: list[pd.DataFrame] = []
    for spec in specs:
        column = spec.column
        if column not in df.columns:
            source = pd.Series([spec.default_value] * len(df), index=df.index)
        else:
            source = df[column]

        if spec.encoding in {"numeric_scaled", "numeric_raw"}:
            numeric = _coerce_numeric(source)
            fill = float(spec.fill_value if spec.fill_value is not None else 0.0)
            filled = numeric.fillna(fill).astype(float)
            if spec.encoding == "numeric_scaled":
                mean = float(spec.mean if spec.mean is not None else 0.0)
                std = float(spec.std if spec.std is not None else 1.0)
                if std <= 0:
                    std = 1.0
                filled = (filled - mean) / std
            encoded_parts.append(pd.DataFrame({spec.feature_names[0]: filled}, index=df.index))
            continue

        if spec.encoding == "one_hot":
            values = _normalise_string(source)
            encoded_map = {
                feature_name: (values == feature_name.split("=", 1)[1]).astype(float)
                for feature_name in spec.feature_names
            }
            part = pd.DataFrame(encoded_map, index=df.index)
            encoded_parts.append(part)
            continue

        if spec.encoding == "ordinal":
            values = _normalise_string(source)
            categories = spec.categories or []
            mapping = {category: float(idx) for idx, category in enumerate(categories)}
            encoded = values.map(mapping).fillna(-1.0).astype(float)
            encoded_parts.append(pd.DataFrame({spec.feature_names[0]: encoded}, index=df.index))
            continue

        parser = _parse_multi_hot_literal
        if spec.encoding == "multi_hot_delimited":
            parser = lambda value: _parse_multi_hot_delimited(value, spec.delimiter)
        tokens_per_row = source.apply(parser)
        encoded_map = {
            feature_name: tokens_per_row.apply(
                lambda values, token=feature_name.split(" contains ", 1)[1]: float(token in values)
            )
            for feature_name in spec.feature_names
        }
        part = pd.DataFrame(encoded_map, index=df.index)
        encoded_parts.append(part)

    if not encoded_parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(encoded_parts, axis=1).astype(float)


def train_dynamic_model(
    df: pd.DataFrame,
    target_column: str,
    feature_specs: list[FeatureEncodingSpec],
    model_key: str = DEFAULT_MODEL_KEY,
    random_state: int = 42,
    test_size: float = 0.25,
    interval_z: float = 1.64,
) -> DynamicModelBundle:
    if not feature_specs:
        raise ValueError("At least one feature specification is required for training.")

    missing = [spec.column for spec in feature_specs if spec.column not in df.columns]
    if missing:
        raise ValueError(f"The following treatment columns are missing: {sorted(set(missing))}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    fitted_specs: list[FittedFeatureSpec] = []
    encoded_parts: list[pd.DataFrame] = []
    for spec in feature_specs:
        fitted, part = _fit_feature_spec(df[spec.column], spec)
        fitted_specs.append(fitted)
        if not part.empty:
            encoded_parts.append(part)

    if not encoded_parts:
        raise ValueError("Selected encodings produced zero usable model features.")
    encoded = pd.concat(encoded_parts, axis=1).astype(float)
    encoded = encoded.loc[:, encoded.var(axis=0, numeric_only=True) > 0]
    if encoded.empty:
        raise ValueError("All encoded features are constant; choose different treatments or encodings.")

    target = _coerce_numeric(df[target_column])
    valid_mask = target.notna()
    encoded_valid = encoded.loc[valid_mask].copy()
    target_valid = target.loc[valid_mask].copy()
    if len(encoded_valid) < 8:
        raise ValueError("Need at least 8 rows with numeric target values for training.")

    holdout_diagnostics = pd.DataFrame(
        columns=["source_index", "actual", "prediction", "std", "lower", "upper", "residual"]
    )

    regressor = _regressor_for_model(model_key=model_key, random_state=random_state)

    if len(encoded_valid) >= 12:
        x_train, x_test, y_train, y_test = train_test_split(
            encoded_valid,
            target_valid,
            test_size=test_size,
            random_state=random_state,
        )
        eval_model = clone(regressor)
        eval_model.fit(x_train, y_train)
        train_pred, train_std, _, _ = _predict_with_uncertainty_matrix(eval_model, x_train, interval_z=interval_z)
        test_pred, test_std, test_lower, test_upper = _predict_with_uncertainty_matrix(
            eval_model,
            x_test,
            interval_z=interval_z,
        )
        holdout_diagnostics = pd.DataFrame(
            {
                "source_index": x_test.index,
                "actual": y_test.to_numpy(),
                "prediction": test_pred,
                "std": test_std,
                "lower": test_lower,
                "upper": test_upper,
            }
        )
        holdout_diagnostics["residual"] = holdout_diagnostics["actual"] - holdout_diagnostics["prediction"]
        metrics = {
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "train_mean_std": float(train_std.mean()),
            "test_mean_std": float(test_std.mean()),
            "train_rows": float(len(x_train)),
            "test_rows": float(len(x_test)),
        }
    else:
        metrics = {
            "train_r2": float("nan"),
            "test_r2": float("nan"),
            "train_mae": float("nan"),
            "test_mae": float("nan"),
            "train_mean_std": float("nan"),
            "test_mean_std": float("nan"),
            "train_rows": float(len(encoded_valid)),
            "test_rows": 0.0,
        }

    final_model = clone(regressor)
    final_model.fit(encoded_valid, target_valid)
    return DynamicModelBundle(
        regressor=final_model,
        target_column=target_column,
        feature_specs=fitted_specs,
        encoded_feature_names=list(encoded_valid.columns),
        metrics=metrics,
        holdout_diagnostics=holdout_diagnostics,
        training_encoded=encoded_valid,
        training_target=target_valid,
        model_key=model_key,
        model_label=MODEL_LABELS[model_key],
        interval_z=interval_z,
    )


def make_dynamic_prediction_frame(raw_values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([raw_values])


def predict_dynamic(bundle: DynamicModelBundle, frame: pd.DataFrame) -> PredictionResult:
    transformed = _transform_with_fitted_specs(frame, bundle.feature_specs)
    for column in bundle.encoded_feature_names:
        if column not in transformed.columns:
            transformed[column] = 0.0
    transformed = transformed[bundle.encoded_feature_names]
    prediction, std, lower, upper = _predict_with_uncertainty_matrix(
        bundle.regressor,
        transformed,
        interval_z=bundle.interval_z,
    )
    value = float(prediction[0])
    uncertainty = float(std[0])
    return PredictionResult(
        prediction=value,
        lower=float(lower[0]),
        upper=float(upper[0]),
        std=uncertainty,
    )
