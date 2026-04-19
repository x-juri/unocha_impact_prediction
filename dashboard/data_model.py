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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CERF_DATA_PATH = PROJECT_ROOT / "Final" / "CERF_UFE_2017_24_CIRV_Match.csv"
CBPF_DATA_PATH = (
    PROJECT_ROOT
    / "Final"
    / "Data_ CERF Donor Contributions and Allocations - allocations - CBPFs Projects.csv"
)
DATA_PATH = CERF_DATA_PATH

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
    model_key: str
    model_label: str
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


def build_cerf_regression_pipeline(
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> Pipeline:
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
            ("regressor", _regressor_for_model(model_key, random_state=random_state)),
        ]
    )


def build_regression_pipeline(
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> Pipeline:
    return build_cerf_regression_pipeline(model_key=model_key, random_state=random_state)


def build_cbpf_regression_pipeline(
    sector_columns: list[str],
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> Pipeline:
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
            ("regressor", _regressor_for_model(model_key, random_state=random_state)),
        ]
    )


def _train_model_bundle(
    df: pd.DataFrame,
    feature_columns: list[str],
    pipeline: Pipeline,
    target_column: str = TARGET_NUMERIC_COLUMN,
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> ModelBundle:
    model_data = df.dropna(subset=[target_column, *feature_columns]).copy()
    if model_data.empty:
        raise ValueError("No complete rows are available for model training.")

    x = model_data[feature_columns]
    y = model_data[target_column]

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
        model_key=model_key,
        model_label=MODEL_LABELS[model_key],
    )


def train_cerf_model(
    df: pd.DataFrame,
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> ModelBundle:
    return _train_model_bundle(
        df=df,
        feature_columns=CERF_MODEL_FEATURES,
        pipeline=build_cerf_regression_pipeline(model_key=model_key, random_state=random_state),
        model_key=model_key,
        random_state=random_state,
    )


def train_model(
    df: pd.DataFrame,
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> ModelBundle:
    return train_cerf_model(df, model_key=model_key, random_state=random_state)


def train_cbpf_model(
    df: pd.DataFrame,
    model_key: str = MODEL_BAYESIAN_RIDGE,
    random_state: int = 42,
) -> ModelBundle:
    sector_columns = cbpf_sector_feature_columns(df)
    return _train_model_bundle(
        df=df,
        feature_columns=[*CBPF_NUMERIC_FEATURES, *CBPF_CATEGORICAL_FEATURES, *sector_columns],
        pipeline=build_cbpf_regression_pipeline(
            sector_columns,
            model_key=model_key,
            random_state=random_state,
        ),
        model_key=model_key,
        random_state=random_state,
    )


def train_cerf_models(
    df: pd.DataFrame,
    model_keys: Iterable[str] = AVAILABLE_MODEL_KEYS,
    random_state: int = 42,
) -> dict[str, ModelBundle]:
    return {
        model_key: train_cerf_model(df, model_key=model_key, random_state=random_state)
        for model_key in model_keys
    }


def train_cbpf_models(
    df: pd.DataFrame,
    model_keys: Iterable[str] = AVAILABLE_MODEL_KEYS,
    random_state: int = 42,
) -> dict[str, ModelBundle]:
    return {
        model_key: train_cbpf_model(df, model_key=model_key, random_state=random_state)
        for model_key in model_keys
    }


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
        if hasattr(regressor, "estimators_"):
            tree_predictions = np.vstack(
                [estimator.predict(transformed) for estimator in regressor.estimators_]
            )
            predictions = regressor.predict(transformed)
            std = tree_predictions.std(axis=0, ddof=1)
            lower = np.percentile(tree_predictions, 5, axis=0)
            upper = np.percentile(tree_predictions, 95, axis=0)
        else:
            predictions, std = regressor.predict(transformed, return_std=True)
            lower = predictions - 1.64 * std
            upper = predictions + 1.64 * std
    return pd.DataFrame(
        {
            "prediction": predictions,
            "std": std,
            "lower": lower,
            "upper": upper,
        }
    )


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
        lower=float(result.get("lower", prediction - bundle.interval_z * std)),
        upper=float(result.get("upper", prediction + bundle.interval_z * std)),
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
    regressor = bundle.pipeline.named_steps["regressor"]
    if not hasattr(regressor, "coef_"):
        raise ValueError(f"{bundle.model_label} does not expose regression coefficients.")
    coefficients = regressor.coef_
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
