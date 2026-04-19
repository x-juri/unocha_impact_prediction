from __future__ import annotations

import math
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

try:
    from .data_model import (
        AMOUNT_NUMERIC_COLUMN,
        CBPF_ALLOCATION_SOURCE_COLUMN,
        CBPF_BUDGET_NUMERIC_COLUMN,
        CBPF_COUNTRY_COLUMN,
        CBPF_DATA_PATH,
        CBPF_DURATION_NUMERIC_COLUMN,
        CBPF_ORGANIZATION_TYPE_COLUMN,
        CBPF_PROJECT_SECTOR_LIST_COLUMN,
        CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
        CBPF_YEAR_COLUMN,
        CERF_DATA_PATH,
        CIRV_PREV_NUMERIC_COLUMN,
        DEFAULT_MODEL_KEY,
        MODEL_BAYESIAN_RIDGE,
        MODEL_OPTIONS,
        MODEL_RANDOM_FOREST,
        TARGET_NUMERIC_COLUMN,
        budget_sensecheck,
        cbpf_sector_feature_columns,
        cbpf_sector_names,
        dropdown_options,
        load_and_clean_cbpf_projects,
        load_and_clean_cerf_data,
        make_cbpf_prediction_frame,
        make_cerf_prediction_frame,
        predict_with_uncertainty,
        train_cbpf_models,
        train_cerf_models,
        year_options,
    )
except ImportError:
    from data_model import (
        AMOUNT_NUMERIC_COLUMN,
        CBPF_ALLOCATION_SOURCE_COLUMN,
        CBPF_BUDGET_NUMERIC_COLUMN,
        CBPF_COUNTRY_COLUMN,
        CBPF_DATA_PATH,
        CBPF_DURATION_NUMERIC_COLUMN,
        CBPF_ORGANIZATION_TYPE_COLUMN,
        CBPF_PROJECT_SECTOR_LIST_COLUMN,
        CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
        CBPF_YEAR_COLUMN,
        CERF_DATA_PATH,
        CIRV_PREV_NUMERIC_COLUMN,
        DEFAULT_MODEL_KEY,
        MODEL_BAYESIAN_RIDGE,
        MODEL_OPTIONS,
        MODEL_RANDOM_FOREST,
        TARGET_NUMERIC_COLUMN,
        budget_sensecheck,
        cbpf_sector_feature_columns,
        cbpf_sector_names,
        dropdown_options,
        load_and_clean_cbpf_projects,
        load_and_clean_cerf_data,
        make_cbpf_prediction_frame,
        make_cerf_prediction_frame,
        predict_with_uncertainty,
        train_cbpf_models,
        train_cerf_models,
        year_options,
    )


CERF_DATA = load_and_clean_cerf_data(CERF_DATA_PATH)
CBPF_DATA = load_and_clean_cbpf_projects(CBPF_DATA_PATH)
CERF_MODELS = train_cerf_models(CERF_DATA)
CBPF_MODELS = train_cbpf_models(CBPF_DATA)
CBPF_SECTOR_COLUMNS = cbpf_sector_feature_columns(CBPF_DATA)

CERF_DEFAULT_AMOUNT = float(CERF_DATA[AMOUNT_NUMERIC_COLUMN].median())
CBPF_DEFAULT_DURATION = float(CBPF_DATA[CBPF_DURATION_NUMERIC_COLUMN].median())
CBPF_DEFAULT_BUDGET = float(CBPF_DATA[CBPF_BUDGET_NUMERIC_COLUMN].median())
CBPF_DEFAULT_PEOPLE = float(CBPF_DATA[CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN].median())
CBPF_DEFAULT_CIRV_PREV = float(CBPF_DATA[CIRV_PREV_NUMERIC_COLUMN].median())

CERF_COUNTRY_OPTIONS = dropdown_options(CERF_DATA["countryName"])
CERF_EMERGENCY_OPTIONS = dropdown_options(CERF_DATA["emergencyTypeName"])
CERF_SECTOR_OPTIONS = dropdown_options(CERF_DATA["projectsectors"])
CERF_YEAR_OPTIONS = year_options(CERF_DATA)

CBPF_COUNTRY_OPTIONS = dropdown_options(CBPF_DATA[CBPF_COUNTRY_COLUMN])
CBPF_ALLOCATION_OPTIONS = dropdown_options(CBPF_DATA[CBPF_ALLOCATION_SOURCE_COLUMN])
CBPF_ORG_OPTIONS = dropdown_options(CBPF_DATA[CBPF_ORGANIZATION_TYPE_COLUMN])
CBPF_SECTOR_OPTIONS = dropdown_options(cbpf_sector_names(CBPF_DATA))
CBPF_YEAR_OPTIONS = year_options(CBPF_DATA, CBPF_YEAR_COLUMN)
CBPF_BUDGET_PROFILE = budget_sensecheck(CBPF_DATA)

COLOR_SEQUENCE = ["#1f7a8c", "#74b3ce", "#f2a65a", "#8f3985", "#4f6f52", "#d45d79"]
LABELS = {
    TARGET_NUMERIC_COLUMN: "CIRV - Inc",
    AMOUNT_NUMERIC_COLUMN: "Total amount approved (USD)",
    CBPF_BUDGET_NUMERIC_COLUMN: "Budget",
    CBPF_DURATION_NUMERIC_COLUMN: "Project duration (months)",
    CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN: "Total people",
    CIRV_PREV_NUMERIC_COLUMN: "CIRV - Prev",
    "countryName": "Country",
    "emergencyTypeName": "Emergency type",
    "projectsectors": "Project sector",
    CBPF_COUNTRY_COLUMN: "Country",
    CBPF_ALLOCATION_SOURCE_COLUMN: "Allocation source",
    CBPF_ORGANIZATION_TYPE_COLUMN: "Organization type",
    "sector": "Project sector",
    "mean_cirv": "Mean CIRV - Inc",
    "count": "Rows",
}


def most_common(df: pd.DataFrame, column: str) -> str | None:
    values = df[column].dropna()
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def select_model(models: dict, model_key: str | None):
    return models.get(model_key or DEFAULT_MODEL_KEY, models[DEFAULT_MODEL_KEY])


def coerce_number(value, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        if isinstance(value, str):
            clean_value = value.strip().replace(",", "")
            if not clean_value:
                return fallback
            number = float(clean_value)
        else:
            number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def format_cirv(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):+.3f}"


def format_number(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):,.0f}"


def format_money(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.0f}"


def metric_card(label: str, value: str, helper: str | None = None) -> html.Div:
    children = [html.Span(label, className="metric-label"), html.Strong(value)]
    if helper:
        children.append(html.Small(helper))
    return html.Div(children, className="metric-card")


def model_diagnostics(bundle, description: str) -> html.Div:
    metrics = bundle.metrics
    return html.Div(
        [
            html.Span("Model diagnostics", className="section-kicker"),
            html.Div(
                [
                    metric_card("Model", bundle.model_label),
                    metric_card("Test R2", f"{metrics['test_r2']:.3f}"),
                    metric_card("Test MAE", f"{metrics['test_mae']:.3f}"),
                    metric_card("Mean test std", f"{metrics['test_mean_std']:.3f}"),
                ],
                className="metrics-grid compact",
            ),
            html.P(description, className="muted"),
        ],
        className="diagnostics",
    )


def model_description(bundle, dataset: str) -> str:
    if bundle.model_key == MODEL_RANDOM_FOREST:
        interval_text = "The interval is the 5th-95th percentile spread across the fitted trees."
    elif bundle.model_key == MODEL_BAYESIAN_RIDGE:
        interval_text = "The interval uses the Bayesian Ridge predictive standard deviation."
    else:
        interval_text = "The interval is model-derived."

    if dataset == "CERF":
        features = "total amount approved plus encoded emergency type, country, and project sector"
    else:
        features = (
            "duration, budget, total people, CIRV - Prev, encoded allocation source, "
            "organization type, and multi-hot project sectors"
        )
    rows = int(bundle.metrics["train_rows"] + bundle.metrics["test_rows"])
    return f"{bundle.model_label} using {features}. {interval_text} Rows used for evaluation: {rows:,}."


def prediction_box(result, model_label: str, label: str = "Estimated CIRV - Inc") -> html.Div:
    return html.Div(
        [
            html.Span(label, className="prediction-label"),
            html.Strong(format_cirv(result.prediction)),
            html.P(f"90% model interval: {format_cirv(result.lower)} to {format_cirv(result.upper)}"),
            html.Small(f"{model_label} uncertainty std: {result.std:.3f}"),
        ],
        className="prediction-box",
    )


def warning_box(title: str, message: str) -> html.Div:
    return html.Div(
        [
            html.Span(title, className="prediction-label"),
            html.P(message),
        ],
        className="prediction-box warning",
    )


def empty_figure(message: str):
    fig = px.scatter(title=message)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return apply_chart_style(fig)


def apply_chart_style(fig):
    fig.update_layout(
        template="plotly_white",
        colorway=COLOR_SEQUENCE,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#1f2933"},
        title={"font": {"size": 18}},
    )
    return fig


def top_values(df: pd.DataFrame, column: str, limit: int) -> list[str]:
    return df[column].value_counts().head(limit).index.tolist()


def filter_cerf_data(years, countries, emergency_types, sectors) -> pd.DataFrame:
    filtered = CERF_DATA
    if years:
        filtered = filtered[filtered["year"].astype(int).isin(years)]
    if countries:
        filtered = filtered[filtered["countryName"].isin(countries)]
    if emergency_types:
        filtered = filtered[filtered["emergencyTypeName"].isin(emergency_types)]
    if sectors:
        filtered = filtered[filtered["projectsectors"].isin(sectors)]
    return filtered.copy()


def filter_cbpf_data(years, countries, allocation_sources, org_types, sectors) -> pd.DataFrame:
    filtered = CBPF_DATA
    if years:
        filtered = filtered[filtered[CBPF_YEAR_COLUMN].astype(int).isin(years)]
    if countries:
        filtered = filtered[filtered[CBPF_COUNTRY_COLUMN].isin(countries)]
    if allocation_sources:
        filtered = filtered[filtered[CBPF_ALLOCATION_SOURCE_COLUMN].isin(allocation_sources)]
    if org_types:
        filtered = filtered[filtered[CBPF_ORGANIZATION_TYPE_COLUMN].isin(org_types)]
    if sectors:
        selected = set(sectors)
        filtered = filtered[
            filtered[CBPF_PROJECT_SECTOR_LIST_COLUMN].apply(lambda values: bool(selected.intersection(values)))
        ]
    return filtered.copy()


def cerf_summary_cards(df: pd.DataFrame) -> list[html.Div]:
    if df.empty:
        return [
            metric_card("Rows", "0", "No records match the active filters"),
            metric_card("Year range", "n/a"),
            metric_card("Mean CIRV - Inc", "n/a"),
            metric_card("Total approved", "n/a"),
        ]

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    year_label = str(min_year) if min_year == max_year else f"{min_year}-{max_year}"
    return [
        metric_card("Rows", f"{len(df):,}", f"{df['countryName'].nunique():,} countries"),
        metric_card("Year range", year_label),
        metric_card(
            "Mean / median CIRV - Inc",
            format_cirv(df[TARGET_NUMERIC_COLUMN].mean()),
            f"Median {format_cirv(df[TARGET_NUMERIC_COLUMN].median())}",
        ),
        metric_card("Total approved", format_money(df[AMOUNT_NUMERIC_COLUMN].sum())),
    ]


def cbpf_summary_cards(df: pd.DataFrame) -> list[html.Div]:
    if df.empty:
        return [
            metric_card("Rows", "0", "No records match the active filters"),
            metric_card("Year range", "n/a"),
            metric_card("Mean CIRV - Inc", "n/a"),
            metric_card("Total budget", "n/a"),
        ]

    min_year = int(df[CBPF_YEAR_COLUMN].min())
    max_year = int(df[CBPF_YEAR_COLUMN].max())
    year_label = str(min_year) if min_year == max_year else f"{min_year}-{max_year}"
    return [
        metric_card("Rows", f"{len(df):,}", f"{df[CBPF_COUNTRY_COLUMN].nunique():,} countries"),
        metric_card("Year range", year_label),
        metric_card(
            "Mean / median CIRV - Inc",
            format_cirv(df[TARGET_NUMERIC_COLUMN].mean()),
            f"Median {format_cirv(df[TARGET_NUMERIC_COLUMN].median())}",
        ),
        metric_card(
            "Total budget / people",
            format_money(df[CBPF_BUDGET_NUMERIC_COLUMN].sum()),
            f"{format_number(df[CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN].sum())} people",
        ),
    ]


def target_distribution(df: pd.DataFrame, title: str):
    if df.empty:
        return empty_figure("No CIRV values for the active filters")
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[TARGET_NUMERIC_COLUMN],
                nbinsx=30,
                marker_color="#1f7a8c",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title=LABELS[TARGET_NUMERIC_COLUMN], yaxis_title="Count")
    return apply_chart_style(fig)


def cerf_amount_scatter(df: pd.DataFrame):
    if df.empty:
        return empty_figure("No amount records for the active filters")
    fig = px.scatter(
        df,
        x=AMOUNT_NUMERIC_COLUMN,
        y=TARGET_NUMERIC_COLUMN,
        color="emergencyTypeName",
        hover_data=["countryName", "year", "projectsectors"],
        labels=LABELS,
        log_x=True,
        title="CERF approved amount vs CIRV - Inc",
    )
    return apply_chart_style(fig)


def cbpf_budget_scatter(df: pd.DataFrame):
    if df.empty:
        return empty_figure("No budget records for the active filters")
    fig = px.scatter(
        df,
        x=CBPF_BUDGET_NUMERIC_COLUMN,
        y=TARGET_NUMERIC_COLUMN,
        color=CBPF_ORGANIZATION_TYPE_COLUMN,
        hover_data=[CBPF_COUNTRY_COLUMN, CBPF_YEAR_COLUMN, CBPF_ALLOCATION_SOURCE_COLUMN],
        labels=LABELS,
        log_x=True,
        title="CBPF budget vs CIRV - Inc",
    )
    return apply_chart_style(fig)


def box_by_category(df: pd.DataFrame, column: str, limit: int, title: str):
    if df.empty:
        return empty_figure("No category records for the active filters")
    values = top_values(df, column, limit)
    chart_df = df[df[column].isin(values)]
    fig = px.box(
        chart_df,
        x=column,
        y=TARGET_NUMERIC_COLUMN,
        points="outliers",
        labels=LABELS,
        title=title,
    )
    fig.update_xaxes(tickangle=35)
    return apply_chart_style(fig)


def mean_bar(df: pd.DataFrame, column: str, limit: int, title: str):
    if df.empty:
        return empty_figure("No category records for the active filters")
    values = top_values(df, column, limit)
    grouped = (
        df[df[column].isin(values)]
        .groupby(column, as_index=False)
        .agg(mean_cirv=(TARGET_NUMERIC_COLUMN, "mean"), count=(TARGET_NUMERIC_COLUMN, "size"))
        .sort_values("mean_cirv")
    )
    fig = px.bar(
        grouped,
        x="mean_cirv",
        y=column,
        orientation="h",
        hover_data=["count"],
        labels=LABELS,
        title=title,
        color_discrete_sequence=["#1f7a8c"],
    )
    fig.update_layout(height=max(360, 32 * len(grouped) + 120))
    return apply_chart_style(fig)


def cbpf_sector_bar(df: pd.DataFrame, limit: int):
    if df.empty:
        return empty_figure("No project sector records for the active filters")
    exploded = (
        df[[TARGET_NUMERIC_COLUMN, CBPF_PROJECT_SECTOR_LIST_COLUMN]]
        .explode(CBPF_PROJECT_SECTOR_LIST_COLUMN)
        .rename(columns={CBPF_PROJECT_SECTOR_LIST_COLUMN: "sector"})
    )
    exploded = exploded[exploded["sector"].notna()]
    values = top_values(exploded, "sector", limit)
    grouped = (
        exploded[exploded["sector"].isin(values)]
        .groupby("sector", as_index=False)
        .agg(mean_cirv=(TARGET_NUMERIC_COLUMN, "mean"), count=(TARGET_NUMERIC_COLUMN, "size"))
        .sort_values("mean_cirv")
    )
    fig = px.bar(
        grouped,
        x="mean_cirv",
        y="sector",
        orientation="h",
        hover_data=["count"],
        labels=LABELS,
        title=f"Mean CIRV - Inc by top {len(grouped)} CBPF project sectors",
        color_discrete_sequence=["#1f7a8c"],
    )
    return apply_chart_style(fig)


def budget_sensecheck_component() -> html.Div:
    examples = ", ".join(CBPF_BUDGET_PROFILE["examples"][:5])
    return html.Div(
        [
            html.Span("Budget parsing check", className="section-kicker"),
            html.Div(
                [
                    metric_card("Raw dtype", str(CBPF_BUDGET_PROFILE["raw_dtype"])),
                    metric_card("Comma values", str(CBPF_BUDGET_PROFILE["raw_values_with_comma"])),
                    metric_card("Parsed missing", str(CBPF_BUDGET_PROFILE["parsed_missing"])),
                    metric_card("Median budget", format_money(CBPF_BUDGET_PROFILE["median"])),
                ],
                className="metrics-grid compact",
            ),
            html.P(
                f"Sample raw values: {examples}. Parsed range: "
                f"{format_money(CBPF_BUDGET_PROFILE['min'])} to {format_money(CBPF_BUDGET_PROFILE['max'])}.",
                className="muted",
            ),
        ],
        className="diagnostics",
    )


app = Dash(__name__, title="CERF and CBPF CIRV Dashboard")
server = app.server

app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P("CERF and CBPF 2017-2024", className="eyebrow"),
                        html.H1("CIRV impact estimator"),
                        html.P(
                            "Estimate CIRV - Inc with selectable Random Forest or Bayesian Ridge models, then explore the historical records behind each dataset.",
                            className="lede",
                        ),
                    ],
                    className="header-copy",
                ),
                html.Div(
                    [
                        html.Span("Data sources"),
                        html.Strong(CERF_DATA_PATH.name),
                        html.Strong(CBPF_DATA_PATH.name),
                        html.Small(f"{len(CERF_DATA):,} CERF rows, {len(CBPF_DATA):,} CBPF rows"),
                    ],
                    className="source-panel",
                ),
            ],
            className="page-header",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.Div(
                            [
                                html.Span("Models", className="section-kicker"),
                                html.H2("Estimate CIRV - Inc"),
                                html.P(
                                    "Random Forest is the default model. Intervals are approximate model uncertainty ranges, not causal uncertainty bounds.",
                                    className="muted",
                                ),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("CERF UFE allocation model"),
                                        html.Div(
                                            [
                                                html.Label("Model"),
                                                dcc.Dropdown(
                                                    id="cerf-model-type",
                                                    options=MODEL_OPTIONS,
                                                    value=DEFAULT_MODEL_KEY,
                                                    clearable=False,
                                                ),
                                                html.Label("Total amount approved"),
                                                dcc.Input(
                                                    id="cerf-model-amount",
                                                    type="number",
                                                    step=10_000,
                                                    value=round(CERF_DEFAULT_AMOUNT, 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Emergency type"),
                                                dcc.Dropdown(
                                                    id="cerf-model-emergency",
                                                    options=CERF_EMERGENCY_OPTIONS,
                                                    value=most_common(CERF_DATA, "emergencyTypeName"),
                                                    clearable=False,
                                                ),
                                                html.Label("Country"),
                                                dcc.Dropdown(
                                                    id="cerf-model-country",
                                                    options=CERF_COUNTRY_OPTIONS,
                                                    value=most_common(CERF_DATA, "countryName"),
                                                    clearable=False,
                                                ),
                                                html.Label("Project sector"),
                                                dcc.Dropdown(
                                                    id="cerf-model-sector",
                                                    options=CERF_SECTOR_OPTIONS,
                                                    value=most_common(CERF_DATA, "projectsectors"),
                                                    clearable=False,
                                                ),
                                                html.Button("Estimate CERF", id="cerf-estimate-button", n_clicks=0),
                                            ],
                                            className="model-form",
                                        ),
                                        html.Div(id="cerf-prediction-output"),
                                        html.Div(
                                            id="cerf-model-diagnostics",
                                            children=model_diagnostics(
                                                CERF_MODELS[DEFAULT_MODEL_KEY],
                                                model_description(CERF_MODELS[DEFAULT_MODEL_KEY], "CERF"),
                                            ),
                                        ),
                                    ],
                                    className="model-card",
                                ),
                                html.Div(
                                    [
                                        html.H3("CBPF project model"),
                                        html.Div(
                                            [
                                                html.Label("Model"),
                                                dcc.Dropdown(
                                                    id="cbpf-model-type",
                                                    options=MODEL_OPTIONS,
                                                    value=DEFAULT_MODEL_KEY,
                                                    clearable=False,
                                                ),
                                                html.Label("Allocation source"),
                                                dcc.Dropdown(
                                                    id="cbpf-model-allocation",
                                                    options=CBPF_ALLOCATION_OPTIONS,
                                                    value=most_common(CBPF_DATA, CBPF_ALLOCATION_SOURCE_COLUMN),
                                                    clearable=False,
                                                ),
                                                html.Label("Organization type"),
                                                dcc.Dropdown(
                                                    id="cbpf-model-org",
                                                    options=CBPF_ORG_OPTIONS,
                                                    value=most_common(CBPF_DATA, CBPF_ORGANIZATION_TYPE_COLUMN),
                                                    clearable=False,
                                                ),
                                                html.Label("Project duration (months)"),
                                                dcc.Input(
                                                    id="cbpf-model-duration",
                                                    type="number",
                                                    step=1,
                                                    value=round(CBPF_DEFAULT_DURATION, 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Budget"),
                                                dcc.Input(
                                                    id="cbpf-model-budget",
                                                    type="number",
                                                    step=1000,
                                                    value=round(CBPF_DEFAULT_BUDGET, 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Total people"),
                                                dcc.Input(
                                                    id="cbpf-model-people",
                                                    type="number",
                                                    step=100,
                                                    value=round(CBPF_DEFAULT_PEOPLE, 2),
                                                    debounce=True,
                                                ),
                                                html.Label("CIRV - Prev"),
                                                dcc.Input(
                                                    id="cbpf-model-prev",
                                                    type="number",
                                                    step=0.1,
                                                    value=round(CBPF_DEFAULT_CIRV_PREV, 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Project sectors"),
                                                dcc.Dropdown(
                                                    id="cbpf-model-sectors",
                                                    options=CBPF_SECTOR_OPTIONS,
                                                    value=[most_common(CBPF_DATA.explode(CBPF_PROJECT_SECTOR_LIST_COLUMN), CBPF_PROJECT_SECTOR_LIST_COLUMN)],
                                                    multi=True,
                                                    placeholder="Select one or more sectors",
                                                ),
                                                html.Button("Estimate CBPF", id="cbpf-estimate-button", n_clicks=0),
                                            ],
                                            className="model-form",
                                        ),
                                        html.Div(id="cbpf-prediction-output"),
                                        html.Div(
                                            id="cbpf-model-diagnostics",
                                            children=model_diagnostics(
                                                CBPF_MODELS[DEFAULT_MODEL_KEY],
                                                model_description(CBPF_MODELS[DEFAULT_MODEL_KEY], "CBPF"),
                                            ),
                                        ),
                                    ],
                                    className="model-card",
                                ),
                            ],
                            className="model-columns",
                        ),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Span("CBPF EDA", className="section-kicker"),
                                html.H2("Explore CBPF projects"),
                                html.P(
                                    "Filter CBPF project records and check the budget parsing used by the model.",
                                    className="muted",
                                ),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                html.Label("Years"),
                                dcc.Dropdown(
                                    id="cbpf-year-filter",
                                    options=CBPF_YEAR_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All years",
                                ),
                                html.Label("Countries"),
                                dcc.Dropdown(
                                    id="cbpf-country-filter",
                                    options=CBPF_COUNTRY_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All countries",
                                ),
                                html.Label("Allocation sources"),
                                dcc.Dropdown(
                                    id="cbpf-allocation-filter",
                                    options=CBPF_ALLOCATION_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All sources",
                                ),
                                html.Label("Organization types"),
                                dcc.Dropdown(
                                    id="cbpf-org-filter",
                                    options=CBPF_ORG_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All organization types",
                                ),
                                html.Label("Project sectors"),
                                dcc.Dropdown(
                                    id="cbpf-sector-filter",
                                    options=CBPF_SECTOR_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All sectors",
                                ),
                            ],
                            className="filter-grid",
                        ),
                        html.Div(id="cbpf-summary-cards", className="metrics-grid"),
                        html.Div(
                            [
                                budget_sensecheck_component(),
                                dcc.Graph(id="cbpf-target-distribution", config={"displayModeBar": False}),
                                dcc.Graph(id="cbpf-budget-scatter", config={"displayModeBar": False}),
                                dcc.Graph(id="cbpf-org-box", config={"displayModeBar": False}),
                                dcc.Graph(id="cbpf-sector-bar", config={"displayModeBar": False}),
                            ],
                            className="chart-grid",
                        ),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Span("CERF EDA", className="section-kicker"),
                                html.H2("Explore CERF UFE allocations"),
                                html.P(
                                    "Filter CERF records and compare CIRV - Inc across amounts, countries, emergency types, and project sectors.",
                                    className="muted",
                                ),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                html.Label("Years"),
                                dcc.Dropdown(
                                    id="cerf-year-filter",
                                    options=CERF_YEAR_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All years",
                                ),
                                html.Label("Countries"),
                                dcc.Dropdown(
                                    id="cerf-country-filter",
                                    options=CERF_COUNTRY_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All countries",
                                ),
                                html.Label("Emergency types"),
                                dcc.Dropdown(
                                    id="cerf-emergency-filter",
                                    options=CERF_EMERGENCY_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All emergency types",
                                ),
                                html.Label("Project sectors"),
                                dcc.Dropdown(
                                    id="cerf-sector-filter",
                                    options=CERF_SECTOR_OPTIONS,
                                    value=[],
                                    multi=True,
                                    placeholder="All project sectors",
                                ),
                                html.Label("Top categories shown"),
                                dcc.Dropdown(
                                    id="cerf-top-n",
                                    options=[
                                        {"label": "Top 10", "value": 10},
                                        {"label": "Top 15", "value": 15},
                                        {"label": "Top 25", "value": 25},
                                        {"label": "Top 50", "value": 50},
                                    ],
                                    value=15,
                                    clearable=False,
                                ),
                            ],
                            className="filter-grid",
                        ),
                        html.Div(id="cerf-summary-cards", className="metrics-grid"),
                        html.Div(
                            [
                                dcc.Graph(id="cerf-target-distribution", config={"displayModeBar": False}),
                                dcc.Graph(id="cerf-amount-scatter", config={"displayModeBar": False}),
                                dcc.Graph(id="cerf-emergency-box", config={"displayModeBar": False}),
                                dcc.Graph(id="cerf-country-bar", config={"displayModeBar": False}),
                                dcc.Graph(id="cerf-sector-bar", config={"displayModeBar": False}),
                            ],
                            className="chart-grid",
                        ),
                    ],
                    className="panel",
                ),
            ]
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("cerf-model-diagnostics", "children"),
    Input("cerf-model-type", "value"),
)
def update_cerf_model_diagnostics(model_key):
    bundle = select_model(CERF_MODELS, model_key)
    return model_diagnostics(bundle, model_description(bundle, "CERF"))


@app.callback(
    Output("cbpf-model-diagnostics", "children"),
    Input("cbpf-model-type", "value"),
)
def update_cbpf_model_diagnostics(model_key):
    bundle = select_model(CBPF_MODELS, model_key)
    return model_diagnostics(bundle, model_description(bundle, "CBPF"))


@app.callback(
    Output("cerf-prediction-output", "children"),
    Input("cerf-estimate-button", "n_clicks"),
    State("cerf-model-type", "value"),
    State("cerf-model-amount", "value"),
    State("cerf-model-emergency", "value"),
    State("cerf-model-country", "value"),
    State("cerf-model-sector", "value"),
)
def update_cerf_prediction(_n_clicks, model_key, amount, emergency_type, country, sector):
    missing = []
    if emergency_type is None:
        missing.append("emergency type")
    if country is None:
        missing.append("country")
    if sector is None:
        missing.append("project sector")
    if missing:
        return warning_box("Input needed", f"Add {', '.join(missing)} to estimate CIRV - Inc.")

    bundle = select_model(CERF_MODELS, model_key)
    amount_value = coerce_number(amount, CERF_DEFAULT_AMOUNT)
    frame = make_cerf_prediction_frame(amount_value, str(emergency_type), str(country), str(sector))
    return prediction_box(predict_with_uncertainty(bundle, frame), bundle.model_label)


@app.callback(
    Output("cbpf-prediction-output", "children"),
    Input("cbpf-estimate-button", "n_clicks"),
    State("cbpf-model-type", "value"),
    State("cbpf-model-allocation", "value"),
    State("cbpf-model-org", "value"),
    State("cbpf-model-duration", "value"),
    State("cbpf-model-budget", "value"),
    State("cbpf-model-people", "value"),
    State("cbpf-model-prev", "value"),
    State("cbpf-model-sectors", "value"),
)
def update_cbpf_prediction(
    _n_clicks,
    model_key,
    allocation_source,
    organization_type,
    duration,
    budget,
    people,
    cirv_prev,
    sectors,
):
    missing = []
    for value, label in [
        (allocation_source, "allocation source"),
        (organization_type, "organization type"),
    ]:
        if value is None:
            missing.append(label)
    if missing:
        return warning_box("Input needed", f"Add {', '.join(missing)} to estimate CIRV - Inc.")

    bundle = select_model(CBPF_MODELS, model_key)
    selected_sectors = [str(sector) for sector in (sectors or [])]

    frame = make_cbpf_prediction_frame(
        CBPF_SECTOR_COLUMNS,
        allocation_source=str(allocation_source),
        organization_type=str(organization_type),
        project_duration_months=coerce_number(duration, CBPF_DEFAULT_DURATION),
        budget=coerce_number(budget, CBPF_DEFAULT_BUDGET),
        total_people=coerce_number(people, CBPF_DEFAULT_PEOPLE),
        cirv_prev=coerce_number(cirv_prev, CBPF_DEFAULT_CIRV_PREV),
        project_sectors=selected_sectors,
    )
    return prediction_box(predict_with_uncertainty(bundle, frame), bundle.model_label)


@app.callback(
    Output("cbpf-summary-cards", "children"),
    Output("cbpf-target-distribution", "figure"),
    Output("cbpf-budget-scatter", "figure"),
    Output("cbpf-org-box", "figure"),
    Output("cbpf-sector-bar", "figure"),
    Input("cbpf-year-filter", "value"),
    Input("cbpf-country-filter", "value"),
    Input("cbpf-allocation-filter", "value"),
    Input("cbpf-org-filter", "value"),
    Input("cbpf-sector-filter", "value"),
)
def update_cbpf_eda(years, countries, allocation_sources, org_types, sectors):
    filtered = filter_cbpf_data(years, countries, allocation_sources, org_types, sectors)
    return (
        cbpf_summary_cards(filtered),
        target_distribution(filtered, "CBPF CIRV - Inc distribution"),
        cbpf_budget_scatter(filtered),
        box_by_category(
            filtered,
            CBPF_ORGANIZATION_TYPE_COLUMN,
            10,
            "CIRV - Inc by CBPF organization type",
        ),
        cbpf_sector_bar(filtered, 15),
    )


@app.callback(
    Output("cerf-summary-cards", "children"),
    Output("cerf-target-distribution", "figure"),
    Output("cerf-amount-scatter", "figure"),
    Output("cerf-emergency-box", "figure"),
    Output("cerf-country-bar", "figure"),
    Output("cerf-sector-bar", "figure"),
    Input("cerf-year-filter", "value"),
    Input("cerf-country-filter", "value"),
    Input("cerf-emergency-filter", "value"),
    Input("cerf-sector-filter", "value"),
    Input("cerf-top-n", "value"),
)
def update_cerf_eda(years, countries, emergency_types, sectors, top_n):
    filtered = filter_cerf_data(years, countries, emergency_types, sectors)
    limit = int(top_n or 15)
    return (
        cerf_summary_cards(filtered),
        target_distribution(filtered, "CERF CIRV - Inc distribution"),
        cerf_amount_scatter(filtered),
        box_by_category(
            filtered,
            "emergencyTypeName",
            limit,
            f"CIRV - Inc by top {limit} CERF emergency types",
        ),
        mean_bar(filtered, "countryName", limit, f"Mean CIRV - Inc by top {limit} CERF countries"),
        mean_bar(
            filtered,
            "projectsectors",
            limit,
            f"Mean CIRV - Inc by top {limit} CERF project sectors",
        ),
    )


if __name__ == "__main__":
    debug_enabled = os.getenv("DASH_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)
