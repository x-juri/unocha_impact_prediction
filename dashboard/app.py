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
        train_cbpf_model,
        train_cerf_model,
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
        train_cbpf_model,
        train_cerf_model,
        year_options,
    )


CERF_DATA = load_and_clean_cerf_data(CERF_DATA_PATH)
CBPF_DATA = load_and_clean_cbpf_projects(CBPF_DATA_PATH)
CERF_MODEL = train_cerf_model(CERF_DATA)
CBPF_MODEL = train_cbpf_model(CBPF_DATA)
CBPF_SECTOR_COLUMNS = cbpf_sector_feature_columns(CBPF_DATA)

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
CORRELATION_METHOD_OPTIONS = [
    {"label": "Spearman", "value": "spearman"},
    {"label": "Pearson", "value": "pearson"},
]
CORRELATION_TOP_N_OPTIONS = [
    {"label": "Top 5", "value": 5},
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 30", "value": 30},
    {"label": "Top 50", "value": 50},
]


def most_common(df: pd.DataFrame, column: str) -> str | None:
    values = df[column].dropna()
    if values.empty:
        return None
    return str(values.mode().iloc[0])


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
                    metric_card("Test R2", f"{metrics['test_r2']:.3f}"),
                    metric_card("Test MAE", f"{metrics['test_mae']:.3f}"),
                    metric_card("Mean test std", f"{metrics['test_mean_std']:.3f}"),
                    metric_card("Rows", f"{int(metrics['train_rows'] + metrics['test_rows']):,}"),
                ],
                className="metrics-grid compact",
            ),
            html.P(description, className="muted"),
        ],
        className="diagnostics",
    )


def prediction_box(result, label: str = "Estimated CIRV - Inc") -> html.Div:
    return html.Div(
        [
            html.Span(label, className="prediction-label"),
            html.Strong(format_cirv(result.prediction)),
            html.P(f"90% model interval: {format_cirv(result.lower)} to {format_cirv(result.upper)}"),
            html.Small(f"Predictive standard deviation: {result.std:.3f}"),
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
    fig = px.histogram(
        df,
        x=TARGET_NUMERIC_COLUMN,
        nbins=30,
        labels=LABELS,
        title=title,
        color_discrete_sequence=["#1f7a8c"],
    )
    return apply_chart_style(fig)


def _format_feature_name(name: str) -> str:
    replacements = [
        ("countryName_", "Country: "),
        ("emergencyTypeName_", "Emergency type: "),
        ("projectsectors_", "Project sector: "),
        (f"{CBPF_ALLOCATION_SOURCE_COLUMN}_", "Allocation source: "),
        (f"{CBPF_ORGANIZATION_TYPE_COLUMN}_", "Organization type: "),
        ("sector__", "Project sector: "),
    ]
    for prefix, label in replacements:
        if name.startswith(prefix):
            return f"{label}{name.removeprefix(prefix)}"
    return LABELS.get(name, name)


def _prepare_feature_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    passthrough_columns = passthrough_columns or []
    available_numeric = [column for column in numeric_columns if column in df.columns]
    available_categorical = [column for column in categorical_columns if column in df.columns]
    available_passthrough = [column for column in passthrough_columns if column in df.columns]

    numeric_frame = df[available_numeric].apply(pd.to_numeric, errors="coerce")
    encoded = pd.get_dummies(
        df[available_categorical],
        columns=available_categorical,
        prefix=available_categorical,
        drop_first=True,
        dtype=float,
    )
    passthrough = df[available_passthrough].apply(pd.to_numeric, errors="coerce")

    features = pd.concat([numeric_frame, encoded, passthrough], axis=1)
    features = features.loc[:, features.notna().any(axis=0)]
    features = features.fillna(features.median(numeric_only=True))
    if features.empty:
        return features
    variances = features.var(axis=0, numeric_only=True)
    return features.loc[:, variances > 0]


def _feature_target_correlation(
    features: pd.DataFrame,
    target: pd.Series,
    method: str,
    top_n: int,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation", "feature_label"])

    corr_input = features.copy()
    corr_input["__target__"] = pd.to_numeric(target, errors="coerce")
    corr_input = corr_input.dropna(subset=["__target__"])
    if corr_input.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation", "feature_label"])

    correlations = corr_input.corr(method=method, numeric_only=True)["__target__"].drop("__target__").dropna()
    if correlations.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation", "feature_label"])

    selected = correlations.abs().sort_values(ascending=False).head(max(1, int(top_n))).index
    ranked = correlations.loc[selected].sort_values(key=lambda values: values.abs())
    return pd.DataFrame(
        {
            "feature": ranked.index,
            "correlation": ranked.values,
        }
    ).assign(
        abs_correlation=lambda frame: frame["correlation"].abs(),
        feature_label=lambda frame: frame["feature"].map(_format_feature_name),
    )


def correlation_heatmap(
    features: pd.DataFrame,
    top_features: pd.DataFrame,
    method: str,
    title: str,
):
    if features.empty or top_features.empty:
        return empty_figure("Not enough data for pairwise feature correlation")

    selected_columns = [column for column in top_features["feature"] if column in features.columns]
    if len(selected_columns) < 2:
        return empty_figure("Need at least two non-constant features for pairwise correlation")

    corr_matrix = features[selected_columns].corr(method=method, numeric_only=True)
    axis_labels = [_format_feature_name(column) for column in corr_matrix.columns]
    fig = px.imshow(
        corr_matrix,
        x=axis_labels,
        y=axis_labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        origin="lower",
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar={"title": "Corr"}, height=max(380, 48 * len(axis_labels) + 180))
    fig.update_xaxes(tickangle=30)
    return apply_chart_style(fig)


def feature_target_correlation_bar(correlation_df: pd.DataFrame, title: str):
    if correlation_df.empty:
        return empty_figure("Not enough data for feature-to-target correlation")

    fig = px.bar(
        correlation_df,
        x="correlation",
        y="feature_label",
        orientation="h",
        color="correlation",
        labels={"correlation": "Correlation", "feature_label": "Feature"},
        title=title,
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        hover_data={"abs_correlation": ":.3f", "feature": True, "correlation": ":.3f"},
    )
    fig.update_layout(coloraxis_colorbar={"title": "Corr"}, height=max(360, 36 * len(correlation_df) + 120))
    return apply_chart_style(fig)


def _filter_holdout(bundle, filtered: pd.DataFrame) -> pd.DataFrame:
    holdout = bundle.holdout_diagnostics
    if holdout.empty or filtered.empty or "source_index" not in holdout.columns:
        return holdout.iloc[0:0].copy()
    return holdout[holdout["source_index"].isin(filtered.index)].copy()


def _binned_line(frame: pd.DataFrame, x_column: str, bins: int = 20, interval_z: float = 1.64) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = list(dict.fromkeys([x_column, "prediction", "lower", "upper", "residual", "std"]))
    working = frame[columns].dropna().copy()
    if len(working) < 8:
        return pd.DataFrame()
    unique_x = working[x_column].nunique()
    if unique_x < 4:
        return pd.DataFrame()

    bin_count = max(4, min(bins, unique_x))
    working["bin"] = pd.qcut(working[x_column], q=bin_count, duplicates="drop")
    grouped = (
        working.groupby("bin", observed=True, as_index=False)
        .agg(
            x=(x_column, "mean"),
            prediction=("prediction", "mean"),
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            residual=("residual", "mean"),
            std=("std", "mean"),
        )
        .sort_values("x")
    )
    grouped["residual_lower"] = grouped["residual"] - interval_z * grouped["std"]
    grouped["residual_upper"] = grouped["residual"] + interval_z * grouped["std"]
    return grouped


def predicted_vs_actual_ribbon(holdout: pd.DataFrame, title: str, interval_z: float = 1.64):
    if holdout.empty:
        return empty_figure("No holdout rows for the active filters")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holdout["actual"],
            y=holdout["prediction"],
            mode="markers",
            marker={"size": 6, "opacity": 0.35, "color": "#1f7a8c"},
            name="Holdout rows",
            hovertemplate="Actual: %{x:.3f}<br>Prediction: %{y:.3f}<extra></extra>",
        )
    )

    ribbon = _binned_line(holdout, "actual", interval_z=interval_z)
    if not ribbon.empty:
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["lower"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 122, 140, 0.18)",
                name="90% model ribbon",
                hovertemplate="Ribbon: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["prediction"],
                mode="lines",
                line={"color": "#14515d", "width": 2.5},
                name="Mean prediction",
            )
        )

    low = min(holdout["actual"].min(), holdout["prediction"].min())
    high = max(holdout["actual"].max(), holdout["prediction"].max())
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "#9aa5b1", "dash": "dash"},
            name="Perfect calibration",
        )
    )
    fig.update_layout(title=title, xaxis_title="Actual CIRV - Inc", yaxis_title="Predicted CIRV - Inc")
    return apply_chart_style(fig)


def residual_ribbon(holdout: pd.DataFrame, title: str, interval_z: float = 1.64):
    if holdout.empty:
        return empty_figure("No holdout rows for the active filters")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holdout["prediction"],
            y=holdout["residual"],
            mode="markers",
            marker={"size": 6, "opacity": 0.32, "color": "#74b3ce"},
            name="Residuals",
            hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )

    ribbon = _binned_line(holdout, "prediction", interval_z=interval_z)
    if not ribbon.empty:
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["residual_lower"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["residual_upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(242, 166, 90, 0.2)",
                name="90% model ribbon",
                hovertemplate="Ribbon: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["residual"],
                mode="lines",
                line={"color": "#f2a65a", "width": 2.5},
                name="Mean residual",
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#9aa5b1")
    fig.update_layout(title=title, xaxis_title="Predicted CIRV - Inc", yaxis_title="Residual (actual - predicted)")
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
                            "Estimate CIRV - Inc with Bayesian Ridge models, then explore the historical records behind each dataset.",
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
                                    "Intervals are Bayesian Ridge predictive intervals from the model, not causal uncertainty bounds.",
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
                                                html.Label("Total amount approved"),
                                                dcc.Input(
                                                    id="cerf-model-amount",
                                                    type="number",
                                                    min=0,
                                                    step=10_000,
                                                    value=round(float(CERF_DATA[AMOUNT_NUMERIC_COLUMN].median()), 2),
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
                                        model_diagnostics(
                                            CERF_MODEL,
                                            "Bayesian Ridge using total amount approved plus one-hot encoded emergency type, country, and project sector.",
                                        ),
                                    ],
                                    className="model-card",
                                ),
                                html.Div(
                                    [
                                        html.H3("CBPF project model"),
                                        html.Div(
                                            [
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
                                                    min=0,
                                                    step=1,
                                                    value=round(float(CBPF_DATA[CBPF_DURATION_NUMERIC_COLUMN].median()), 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Budget"),
                                                dcc.Input(
                                                    id="cbpf-model-budget",
                                                    type="number",
                                                    min=0,
                                                    step=1000,
                                                    value=round(float(CBPF_DATA[CBPF_BUDGET_NUMERIC_COLUMN].median()), 2),
                                                    debounce=True,
                                                ),
                                                html.Label("Total people"),
                                                dcc.Input(
                                                    id="cbpf-model-people",
                                                    type="number",
                                                    min=0,
                                                    step=100,
                                                    value=round(float(CBPF_DATA[CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN].median()), 2),
                                                    debounce=True,
                                                ),
                                                html.Label("CIRV - Prev"),
                                                dcc.Input(
                                                    id="cbpf-model-prev",
                                                    type="number",
                                                    step=0.1,
                                                    value=round(float(CBPF_DATA[CIRV_PREV_NUMERIC_COLUMN].median()), 2),
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
                                        model_diagnostics(
                                            CBPF_MODEL,
                                            "Bayesian Ridge using scaled numeric fields, dropped-reference one-hot categories, and multi-hot project sectors.",
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
                                html.Label("Correlation method"),
                                dcc.Dropdown(
                                    id="cbpf-corr-method",
                                    options=CORRELATION_METHOD_OPTIONS,
                                    value="spearman",
                                    clearable=False,
                                ),
                                html.Label("Top features (correlation)"),
                                dcc.Dropdown(
                                    id="cbpf-corr-top-n",
                                    options=CORRELATION_TOP_N_OPTIONS,
                                    value=5,
                                    clearable=False,
                                ),
                            ],
                            className="filter-grid",
                        ),
                        html.Div(id="cbpf-summary-cards", className="metrics-grid"),
                        html.Div(
                            [
                                budget_sensecheck_component(),
                                dcc.Graph(id="cbpf-target-distribution", config={"displayModeBar": False}),
                                dcc.Graph(
                                    id="cbpf-feature-target-corr",
                                    config={"displayModeBar": False},
                                    className="chart-span-full",
                                ),
                                dcc.Graph(
                                    id="cbpf-feature-corr-heatmap",
                                    config={"displayModeBar": False},
                                    className="chart-span-full",
                                ),
                                dcc.Graph(id="cbpf-holdout-prediction", config={"displayModeBar": False}),
                                dcc.Graph(id="cbpf-holdout-residual", config={"displayModeBar": False}),
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
                                html.Label("Correlation method"),
                                dcc.Dropdown(
                                    id="cerf-corr-method",
                                    options=CORRELATION_METHOD_OPTIONS,
                                    value="spearman",
                                    clearable=False,
                                ),
                                html.Label("Top features (correlation)"),
                                dcc.Dropdown(
                                    id="cerf-corr-top-n",
                                    options=CORRELATION_TOP_N_OPTIONS,
                                    value=5,
                                    clearable=False,
                                ),
                            ],
                            className="filter-grid",
                        ),
                        html.Div(id="cerf-summary-cards", className="metrics-grid"),
                        html.Div(
                            [
                                dcc.Graph(id="cerf-target-distribution", config={"displayModeBar": False}),
                                dcc.Graph(
                                    id="cerf-feature-target-corr",
                                    config={"displayModeBar": False},
                                    className="chart-span-full",
                                ),
                                dcc.Graph(
                                    id="cerf-feature-corr-heatmap",
                                    config={"displayModeBar": False},
                                    className="chart-span-full",
                                ),
                                dcc.Graph(id="cerf-holdout-prediction", config={"displayModeBar": False}),
                                dcc.Graph(id="cerf-holdout-residual", config={"displayModeBar": False}),
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
    Output("cerf-prediction-output", "children"),
    Input("cerf-estimate-button", "n_clicks"),
    State("cerf-model-amount", "value"),
    State("cerf-model-emergency", "value"),
    State("cerf-model-country", "value"),
    State("cerf-model-sector", "value"),
)
def update_cerf_prediction(_n_clicks, amount, emergency_type, country, sector):
    missing = []
    if amount is None:
        missing.append("total amount approved")
    if emergency_type is None:
        missing.append("emergency type")
    if country is None:
        missing.append("country")
    if sector is None:
        missing.append("project sector")
    if missing:
        return warning_box("Input needed", f"Add {', '.join(missing)} to estimate CIRV - Inc.")
    if float(amount) < 0:
        return warning_box("Check amount", "Total amount approved must be zero or greater.")

    frame = make_cerf_prediction_frame(float(amount), str(emergency_type), str(country), str(sector))
    return prediction_box(predict_with_uncertainty(CERF_MODEL, frame))


@app.callback(
    Output("cbpf-prediction-output", "children"),
    Input("cbpf-estimate-button", "n_clicks"),
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
        (duration, "project duration"),
        (budget, "budget"),
        (people, "total people"),
        (cirv_prev, "CIRV - Prev"),
    ]:
        if value is None:
            missing.append(label)
    if not sectors:
        missing.append("project sectors")
    if missing:
        return warning_box("Input needed", f"Add {', '.join(missing)} to estimate CIRV - Inc.")

    for value, label in [(duration, "Project duration"), (budget, "Budget"), (people, "Total people")]:
        if float(value) < 0:
            return warning_box("Check input", f"{label} must be zero or greater.")

    frame = make_cbpf_prediction_frame(
        CBPF_SECTOR_COLUMNS,
        allocation_source=str(allocation_source),
        organization_type=str(organization_type),
        project_duration_months=float(duration),
        budget=float(budget),
        total_people=float(people),
        cirv_prev=float(cirv_prev),
        project_sectors=[str(sector) for sector in sectors],
    )
    return prediction_box(predict_with_uncertainty(CBPF_MODEL, frame))


@app.callback(
    Output("cbpf-summary-cards", "children"),
    Output("cbpf-target-distribution", "figure"),
    Output("cbpf-feature-target-corr", "figure"),
    Output("cbpf-feature-corr-heatmap", "figure"),
    Output("cbpf-holdout-prediction", "figure"),
    Output("cbpf-holdout-residual", "figure"),
    Input("cbpf-year-filter", "value"),
    Input("cbpf-country-filter", "value"),
    Input("cbpf-allocation-filter", "value"),
    Input("cbpf-org-filter", "value"),
    Input("cbpf-sector-filter", "value"),
    Input("cbpf-corr-method", "value"),
    Input("cbpf-corr-top-n", "value"),
)
def update_cbpf_eda(years, countries, allocation_sources, org_types, sectors, corr_method, corr_top_n):
    filtered = filter_cbpf_data(years, countries, allocation_sources, org_types, sectors)
    method = str(corr_method or "spearman")
    top_n = int(corr_top_n or 5)

    cbpf_features = _prepare_feature_matrix(
        filtered,
        numeric_columns=[
            CBPF_YEAR_COLUMN,
            CBPF_DURATION_NUMERIC_COLUMN,
            CBPF_BUDGET_NUMERIC_COLUMN,
            CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
            CIRV_PREV_NUMERIC_COLUMN,
        ],
        categorical_columns=[CBPF_ALLOCATION_SOURCE_COLUMN, CBPF_ORGANIZATION_TYPE_COLUMN],
        passthrough_columns=CBPF_SECTOR_COLUMNS,
    )
    cbpf_corr = _feature_target_correlation(
        cbpf_features,
        filtered[TARGET_NUMERIC_COLUMN],
        method=method,
        top_n=top_n,
    )
    cbpf_holdout = _filter_holdout(CBPF_MODEL, filtered)

    return (
        cbpf_summary_cards(filtered),
        target_distribution(filtered, "CBPF CIRV - Inc distribution"),
        feature_target_correlation_bar(
            cbpf_corr,
            f"CBPF top {top_n} feature to CIRV correlation ({method.capitalize()})",
        ),
        correlation_heatmap(
            cbpf_features,
            cbpf_corr,
            method,
            f"CBPF pairwise feature correlation ({method.capitalize()}, top {len(cbpf_corr)})",
        ),
        predicted_vs_actual_ribbon(
            cbpf_holdout,
            "CBPF holdout predicted vs actual with 90% model ribbon",
            interval_z=CBPF_MODEL.interval_z,
        ),
        residual_ribbon(
            cbpf_holdout,
            "CBPF holdout residuals with 90% model ribbon",
            interval_z=CBPF_MODEL.interval_z,
        ),
    )


@app.callback(
    Output("cerf-summary-cards", "children"),
    Output("cerf-target-distribution", "figure"),
    Output("cerf-feature-target-corr", "figure"),
    Output("cerf-feature-corr-heatmap", "figure"),
    Output("cerf-holdout-prediction", "figure"),
    Output("cerf-holdout-residual", "figure"),
    Input("cerf-year-filter", "value"),
    Input("cerf-country-filter", "value"),
    Input("cerf-emergency-filter", "value"),
    Input("cerf-sector-filter", "value"),
    Input("cerf-corr-method", "value"),
    Input("cerf-corr-top-n", "value"),
)
def update_cerf_eda(years, countries, emergency_types, sectors, corr_method, corr_top_n):
    filtered = filter_cerf_data(years, countries, emergency_types, sectors)
    method = str(corr_method or "spearman")
    corr_limit = int(corr_top_n or 5)

    cerf_features = _prepare_feature_matrix(
        filtered,
        numeric_columns=["year", AMOUNT_NUMERIC_COLUMN, CIRV_PREV_NUMERIC_COLUMN],
        categorical_columns=["emergencyTypeName", "countryName", "projectsectors"],
    )
    cerf_corr = _feature_target_correlation(
        cerf_features,
        filtered[TARGET_NUMERIC_COLUMN],
        method=method,
        top_n=corr_limit,
    )
    cerf_holdout = _filter_holdout(CERF_MODEL, filtered)

    return (
        cerf_summary_cards(filtered),
        target_distribution(filtered, "CERF CIRV - Inc distribution"),
        feature_target_correlation_bar(
            cerf_corr,
            f"CERF top {corr_limit} feature to CIRV correlation ({method.capitalize()})",
        ),
        correlation_heatmap(
            cerf_features,
            cerf_corr,
            method,
            f"CERF pairwise feature correlation ({method.capitalize()}, top {len(cerf_corr)})",
        ),
        predicted_vs_actual_ribbon(
            cerf_holdout,
            "CERF holdout predicted vs actual with 90% model ribbon",
            interval_z=CERF_MODEL.interval_z,
        ),
        residual_ribbon(
            cerf_holdout,
            "CERF holdout residuals with 90% model ribbon",
            interval_z=CERF_MODEL.interval_z,
        ),
    )


if __name__ == "__main__":
    debug_enabled = os.getenv("DASH_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)
