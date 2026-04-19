from __future__ import annotations

import base64
import io
import math
import os
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, dash_table

try:
    from .data_model import (
        DynamicModelBundle,
        FeatureEncodingSpec,
        default_encoding_for_series,
        infer_column_roles,
        make_dynamic_prediction_frame,
        predict_dynamic,
        train_dynamic_model,
        validate_training_setup,
    )
except ImportError:
    from data_model import (
        DynamicModelBundle,
        FeatureEncodingSpec,
        default_encoding_for_series,
        infer_column_roles,
        make_dynamic_prediction_frame,
        predict_dynamic,
        train_dynamic_model,
        validate_training_setup,
    )


DATASET_CACHE: dict[str, pd.DataFrame] = {}
MODEL_CACHE: dict[str, DynamicModelBundle] = {}
MAX_CACHE_ITEMS = 8

COLOR_SEQUENCE = ["#1f7a8c", "#74b3ce", "#f2a65a", "#8f3985", "#4f6f52", "#d45d79"]
CORRELATION_METHOD_OPTIONS = [
    {"label": "Spearman", "value": "spearman"},
    {"label": "Pearson", "value": "pearson"},
]
CORRELATION_TOP_N_OPTIONS = [
    {"label": "Top 5", "value": 5},
    {"label": "Top 10", "value": 10},
    {"label": "Top 20", "value": 20},
    {"label": "Top 30", "value": 30},
]
ENCODING_OPTIONS = [
    {"label": "Numeric (scaled)", "value": "numeric_scaled"},
    {"label": "Numeric (raw)", "value": "numeric_raw"},
    {"label": "One-hot", "value": "one_hot"},
    {"label": "Ordinal", "value": "ordinal"},
    {"label": "Multi-hot (literal list)", "value": "multi_hot_literal"},
    {"label": "Multi-hot (delimiter)", "value": "multi_hot_delimited"},
]


def _cache_put(cache: dict[str, Any], value: Any) -> str:
    key = uuid.uuid4().hex
    cache[key] = value
    while len(cache) > MAX_CACHE_ITEMS:
        oldest = next(iter(cache))
        del cache[oldest]
    return key


def _cache_get(cache: dict[str, Any], key: str | None):
    if not key:
        return None
    return cache.get(key)


def _decode_upload(contents: str) -> pd.DataFrame:
    if not contents:
        raise ValueError("Upload payload is empty.")
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    buffer = io.StringIO(decoded.decode("utf-8-sig"))
    return pd.read_csv(buffer)


def _pick_default_target(df: pd.DataFrame, roles: dict[str, str]) -> str:
    numeric_columns = [column for column, role in roles.items() if role == "numeric"]
    if not numeric_columns:
        return str(df.columns[0])

    def score(column: str) -> int:
        label = column.lower()
        value = 0
        if "target" in label or "label" in label or "outcome" in label:
            value += 6
        if "cirv" in label or "civr" in label:
            value += 5
        if "inc" in label:
            value += 3
        if "prev" in label:
            value -= 2
        return value

    ranked = sorted(numeric_columns, key=lambda column: (score(column), column), reverse=True)
    if score(ranked[0]) > 0:
        return ranked[0]
    return numeric_columns[0]


def format_target_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    abs_value = abs(numeric)
    if abs_value >= 1_000_000:
        return f"{numeric:,.0f}"
    if abs_value >= 1_000:
        return f"{numeric:,.2f}"
    if abs_value >= 1:
        return f"{numeric:,.3f}"
    return f"{numeric:,.4f}"


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:,.0f}"


def metric_card(label: str, value: str, helper: str | None = None) -> html.Div:
    children = [html.Span(label, className="metric-label"), html.Strong(value)]
    if helper:
        children.append(html.Small(helper))
    return html.Div(children, className="metric-card")


def warning_box(title: str, message: str) -> html.Div:
    return html.Div(
        [html.Span(title, className="prediction-label"), html.P(message)],
        className="prediction-box warning",
    )


def prediction_box(prediction, target_label: str = "Target") -> html.Div:
    return html.Div(
        [
            html.Span(f"Estimated {target_label}", className="prediction-label"),
            html.Strong(format_target_value(prediction.prediction)),
            html.P(
                f"90% model interval: {format_target_value(prediction.lower)}"
                f" to {format_target_value(prediction.upper)}"
            ),
            html.Small(f"Predictive standard deviation: {format_target_value(prediction.std)}"),
        ],
        className="prediction-box",
    )


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


def _feature_target_correlation(encoded: pd.DataFrame, target: pd.Series, method: str, top_n: int) -> pd.DataFrame:
    if encoded.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])
    corr_input = encoded.copy()
    corr_input["__target__"] = pd.to_numeric(target, errors="coerce")
    corr_input = corr_input.dropna(subset=["__target__"])
    if corr_input.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])
    correlations = corr_input.corr(method=method, numeric_only=True)["__target__"].drop("__target__").dropna()
    if correlations.empty:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])
    selected = correlations.abs().sort_values(ascending=False).head(max(1, int(top_n))).index
    ranked = correlations.loc[selected].sort_values(key=lambda values: values.abs())
    return pd.DataFrame(
        {
            "feature": ranked.index,
            "correlation": ranked.values,
        }
    ).assign(abs_correlation=lambda frame: frame["correlation"].abs())


def feature_target_correlation_bar(correlation_df: pd.DataFrame, title: str):
    if correlation_df.empty:
        return empty_figure("Not enough data for feature-to-target correlation")
    fig = px.bar(
        correlation_df,
        x="correlation",
        y="feature",
        orientation="h",
        color="correlation",
        labels={"correlation": "Correlation", "feature": "Feature"},
        title=title,
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
    )
    fig.update_layout(coloraxis_colorbar={"title": "Corr"}, height=max(360, 30 * len(correlation_df) + 120))
    return apply_chart_style(fig)


def correlation_heatmap(encoded: pd.DataFrame, top_features: pd.DataFrame, method: str, title: str):
    if encoded.empty or top_features.empty:
        return empty_figure("Not enough data for pairwise feature correlation")
    selected = [column for column in top_features["feature"] if column in encoded.columns]
    if len(selected) < 2:
        return empty_figure("Need at least two non-constant features for pairwise correlation")
    matrix = encoded[selected].corr(method=method, numeric_only=True)
    fig = px.imshow(
        matrix,
        x=matrix.columns,
        y=matrix.columns,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        origin="lower",
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar={"title": "Corr"}, height=max(420, 40 * len(selected) + 180))
    fig.update_xaxes(tickangle=30)
    return apply_chart_style(fig)


def _binned_line(frame: pd.DataFrame, x_column: str, interval_z: float = 1.64, bins: int = 20) -> pd.DataFrame:
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


def predicted_vs_actual_ribbon(
    holdout: pd.DataFrame,
    title: str,
    interval_z: float,
    target_label: str,
):
    if holdout.empty:
        return empty_figure("No holdout rows are available for diagnostics")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holdout["actual"],
            y=holdout["prediction"],
            mode="markers",
            marker={"size": 6, "opacity": 0.35, "color": "#1f7a8c"},
            name="Holdout rows",
        )
    )
    ribbon = _binned_line(holdout, "actual", interval_z=interval_z)
    if not ribbon.empty:
        fig.add_trace(go.Scatter(x=ribbon["x"], y=ribbon["lower"], mode="lines", line={"width": 0}, showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=ribbon["x"],
                y=ribbon["upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 122, 140, 0.18)",
                name="90% model ribbon",
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
    fig.update_layout(
        title=title,
        xaxis_title=f"Actual {target_label}",
        yaxis_title=f"Predicted {target_label}",
    )
    return apply_chart_style(fig)


def residual_ribbon(
    holdout: pd.DataFrame,
    title: str,
    interval_z: float,
    target_label: str,
):
    if holdout.empty:
        return empty_figure("No holdout rows are available for diagnostics")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holdout["prediction"],
            y=holdout["residual"],
            mode="markers",
            marker={"size": 6, "opacity": 0.32, "color": "#74b3ce"},
            name="Residuals",
        )
    )
    ribbon = _binned_line(holdout, "prediction", interval_z=interval_z)
    if not ribbon.empty:
        fig.add_trace(
            go.Scatter(x=ribbon["x"], y=ribbon["residual_lower"], mode="lines", line={"width": 0}, showlegend=False)
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
    fig.update_layout(
        title=title,
        xaxis_title=f"Predicted {target_label}",
        yaxis_title=f"Residual ({target_label} actual - predicted)",
    )
    return apply_chart_style(fig)


def target_distribution(series: pd.Series, title: str, target_label: str):
    numeric = series.dropna()
    if numeric.empty:
        return empty_figure("Target could not be parsed as numeric values")
    fig = px.histogram(
        x=numeric,
        nbins=30,
        title=title,
        labels={"x": target_label},
        color_discrete_sequence=["#1f7a8c"],
    )
    return apply_chart_style(fig)


app = Dash(__name__, title="On-the-go CIRV-style modeling")
server = app.server


app.layout = html.Div(
    [
        dcc.Store(id="session-id", data=str(uuid.uuid4())),
        dcc.Store(id="dataset-key"),
        dcc.Store(id="model-key"),
        html.Header(
            [
                html.Div(
                    [
                        html.P("Explorative model training", className="eyebrow"),
                        html.H1("CSV impact estimator"),
                        html.P(
                            "Upload any CSV, configure feature encodings, train on the fly, and run inference with model ribbons.",
                            className="lede",
                        ),
                    ],
                    className="header-copy",
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
                                html.Span("Data", className="section-kicker"),
                                html.H2("1) Upload CSV"),
                            ],
                            className="section-heading",
                        ),
                        dcc.Upload(
                            id="csv-upload",
                            children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
                            className="upload-box",
                            multiple=False,
                        ),
                        html.Div(id="upload-status"),
                        html.Div(id="dataset-summary", className="metrics-grid"),
                        html.Div(id="dataset-head-block"),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Span("Setup", className="section-kicker"),
                                html.H2("2) Choose target, treatments, and encodings"),
                            ],
                            className="section-heading",
                        ),
                        html.Div(
                            [
                                html.Div([html.Label("Target column"), dcc.Dropdown(id="target-column")]),
                                html.Div([html.Label("Treatment variables"), dcc.Dropdown(id="treatment-columns", multi=True)]),
                                html.Div(
                                    [
                                        html.Label("Correlation method"),
                                        dcc.Dropdown(
                                            id="corr-method",
                                            options=CORRELATION_METHOD_OPTIONS,
                                            value="spearman",
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Top features"),
                                        dcc.Dropdown(
                                            id="corr-top-n",
                                            options=CORRELATION_TOP_N_OPTIONS,
                                            value=5,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                            ],
                            className="filter-grid",
                        ),
                        html.Div(id="encoding-config"),
                        html.Button("Train model", id="train-button", n_clicks=0),
                        html.Div(id="train-status"),
                    ],
                    className="panel",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Span("Diagnostics", className="section-kicker"),
                                html.H2("3) Evaluate model"),
                            ],
                            className="section-heading",
                        ),
                        html.Div(id="model-metrics", className="metrics-grid"),
                        html.Div(
                            [
                                dcc.Graph(id="target-distribution", config={"displayModeBar": False}),
                                dcc.Graph(id="feature-target-corr", config={"displayModeBar": False}),
                                dcc.Graph(
                                    id="feature-corr-heatmap",
                                    config={"displayModeBar": False},
                                    className="chart-span-full",
                                ),
                                dcc.Graph(id="holdout-prediction", config={"displayModeBar": False}),
                                dcc.Graph(id="holdout-residual", config={"displayModeBar": False}),
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
                                html.Span("Inference", className="section-kicker"),
                                html.H2("4) Estimate target"),
                            ],
                            className="section-heading",
                        ),
                        html.Div(id="inference-form", className="model-form"),
                        html.Button("Estimate", id="estimate-button", n_clicks=0),
                        html.Div(id="prediction-output"),
                    ],
                    className="panel",
                ),
            ]
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("dataset-key", "data"),
    Output("target-column", "options"),
    Output("target-column", "value"),
    Output("treatment-columns", "options"),
    Output("upload-status", "children"),
    Output("dataset-summary", "children"),
    Output("dataset-head-block", "children"),
    Input("csv-upload", "contents"),
    State("csv-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    try:
        df = _decode_upload(contents)
    except Exception as exc:
        message = warning_box("Upload failed", f"Could not parse CSV: {exc}")
        return None, [], None, [], message, [], []

    if df.empty:
        return None, [], None, [], warning_box("Upload failed", "The uploaded CSV has no rows."), [], []

    dataset_key = _cache_put(DATASET_CACHE, df)
    options = [{"label": column, "value": column} for column in df.columns]
    roles = infer_column_roles(df)
    target_default = _pick_default_target(df, roles)

    summary = [
        metric_card("File", str(filename or "uploaded.csv")),
        metric_card("Rows", format_number(len(df))),
        metric_card("Columns", format_number(len(df.columns))),
        metric_card(
            "Missing cells",
            format_number(int(df.isna().sum().sum())),
            f"{(float(df.isna().sum().sum()) / max(1, df.size)):.1%} of all cells",
        ),
    ]
    status = html.Div(
        [
            html.Span("Upload complete", className="section-kicker"),
            html.P("Select target and treatment variables, then configure encoding per variable.", className="muted"),
        ],
        className="diagnostics",
    )

    preview = df.head(5).copy()
    max_columns = 20
    truncated = len(preview.columns) > max_columns
    preview = preview.iloc[:, :max_columns]
    preview = preview.astype("string").replace("<NA>", "")
    preview_columns = [{"name": column, "id": column} for column in preview.columns]
    preview_block = html.Div(
        [
            html.Span("Data preview", className="section-kicker"),
            html.P("First 5 rows from the uploaded file.", className="muted"),
            html.Small(
                f"Showing first {max_columns} columns for readability."
                if truncated
                else "",
                className="muted",
            ),
            html.Div(
                dash_table.DataTable(
                    columns=preview_columns,
                    data=preview.to_dict("records"),
                    page_action="none",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "minWidth": "120px",
                        "maxWidth": "280px",
                        "whiteSpace": "normal",
                        "fontFamily": "Arial, sans-serif",
                        "fontSize": "12px",
                    },
                    style_header={"fontWeight": "700", "backgroundColor": "#f5f8fa"},
                ),
                className="dataset-head-table-wrap",
            ),
        ],
        className="diagnostics dataset-head-block",
    )
    return dataset_key, options, target_default, options, status, summary, preview_block


@app.callback(
    Output("encoding-config", "children"),
    Input("dataset-key", "data"),
    Input("treatment-columns", "value"),
)
def render_encoding_config(dataset_key, treatments):
    df = _cache_get(DATASET_CACHE, dataset_key)
    if df is None or not treatments:
        return html.P("Choose one or more treatment variables to configure encodings.", className="muted")

    roles = infer_column_roles(df)
    rows = []
    for column in treatments:
        if column not in df.columns:
            continue
        default_spec = default_encoding_for_series(df[column])
        role = roles.get(column, "categorical")
        delimiter_default = default_spec.delimiter or ";"
        rows.append(
            html.Div(
                [
                    html.Div([html.Strong(column), html.Small(f"Detected type: {role}", className="muted")]),
                    dcc.Dropdown(
                        id={"type": "encoding-select", "column": column},
                        options=ENCODING_OPTIONS,
                        value=default_spec.encoding,
                        clearable=False,
                    ),
                    dcc.Input(
                        id={"type": "encoding-delimiter", "column": column},
                        type="text",
                        value=delimiter_default,
                        placeholder="Delimiter for multi-hot",
                        maxLength=3,
                    ),
                ],
                className="encoding-row",
            )
        )
    return html.Div(rows, className="encoding-grid")


def _build_feature_specs(treatments, encodings, delimiters) -> list[FeatureEncodingSpec]:
    specs: list[FeatureEncodingSpec] = []
    if not treatments:
        return specs
    for idx, column in enumerate(treatments):
        encoding = encodings[idx] if idx < len(encodings) and encodings[idx] else "one_hot"
        delimiter = delimiters[idx] if idx < len(delimiters) and delimiters[idx] else None
        if encoding != "multi_hot_delimited":
            delimiter = None
        specs.append(FeatureEncodingSpec(column=column, encoding=encoding, delimiter=delimiter))
    return specs


@app.callback(
    Output("model-key", "data"),
    Output("train-status", "children"),
    Output("model-metrics", "children"),
    Output("target-distribution", "figure"),
    Output("feature-target-corr", "figure"),
    Output("feature-corr-heatmap", "figure"),
    Output("holdout-prediction", "figure"),
    Output("holdout-residual", "figure"),
    Input("train-button", "n_clicks"),
    State("dataset-key", "data"),
    State("target-column", "value"),
    State("treatment-columns", "value"),
    State({"type": "encoding-select", "column": ALL}, "value"),
    State({"type": "encoding-delimiter", "column": ALL}, "value"),
    State("corr-method", "value"),
    State("corr-top-n", "value"),
    prevent_initial_call=True,
)
def train_model(
    _n_clicks,
    dataset_key,
    target_column,
    treatments,
    encoding_values,
    delimiter_values,
    corr_method,
    corr_top_n,
):
    df = _cache_get(DATASET_CACHE, dataset_key)
    if df is None:
        return (
            None,
            warning_box("Training blocked", "Upload a CSV first."),
            [],
            empty_figure("Upload data before training"),
            empty_figure("Upload data before training"),
            empty_figure("Upload data before training"),
            empty_figure("Upload data before training"),
            empty_figure("Upload data before training"),
        )

    errors = validate_training_setup(df, target_column, treatments)
    if errors:
        return (
            None,
            warning_box("Training blocked", " ".join(errors)),
            [],
            empty_figure("Fix setup errors before training"),
            empty_figure("Fix setup errors before training"),
            empty_figure("Fix setup errors before training"),
            empty_figure("Fix setup errors before training"),
            empty_figure("Fix setup errors before training"),
        )

    specs = _build_feature_specs(treatments or [], encoding_values or [], delimiter_values or [])
    try:
        bundle = train_dynamic_model(df, str(target_column), specs)
    except Exception as exc:
        return (
            None,
            warning_box("Training failed", str(exc)),
            [],
            empty_figure("Training failed"),
            empty_figure("Training failed"),
            empty_figure("Training failed"),
            empty_figure("Training failed"),
            empty_figure("Training failed"),
        )

    model_key = _cache_put(MODEL_CACHE, bundle)
    method = str(corr_method or "spearman")
    top_n = int(corr_top_n or 5)
    corr_df = _feature_target_correlation(bundle.training_encoded, bundle.training_target, method, top_n)

    metrics = bundle.metrics
    cards = [
        metric_card("Test R2", f"{metrics['test_r2']:.3f}"),
        metric_card("Test MAE", format_target_value(metrics["test_mae"])),
        metric_card("Mean test std", format_target_value(metrics["test_mean_std"])),
        metric_card("Rows", format_number(metrics["train_rows"] + metrics["test_rows"])),
    ]

    status = html.Div(
        [
            html.Span("Model ready", className="section-kicker"),
            html.P("Training complete. You can inspect diagnostics and run inference below.", className="muted"),
        ],
        className="diagnostics",
    )

    return (
        model_key,
        status,
        cards,
        target_distribution(
            bundle.training_target,
            f"{bundle.target_column} distribution (training rows)",
            bundle.target_column,
        ),
        feature_target_correlation_bar(corr_df, f"Top {top_n} feature correlations ({method.capitalize()})"),
        correlation_heatmap(
            bundle.training_encoded,
            corr_df,
            method,
            f"Pairwise feature correlation ({method.capitalize()}, top {len(corr_df)})",
        ),
        predicted_vs_actual_ribbon(
            bundle.holdout_diagnostics,
            f"Holdout predicted vs actual ({bundle.target_column})",
            bundle.interval_z,
            bundle.target_column,
        ),
        residual_ribbon(
            bundle.holdout_diagnostics,
            f"Holdout residuals ({bundle.target_column})",
            bundle.interval_z,
            bundle.target_column,
        ),
    )


@app.callback(
    Output("inference-form", "children"),
    Input("model-key", "data"),
)
def render_inference_form(model_key):
    bundle = _cache_get(MODEL_CACHE, model_key)
    if bundle is None:
        return html.P("Train a model to unlock inference inputs.", className="muted")

    components = []
    for spec in bundle.feature_specs:
        field_id = {"type": "infer-input", "column": spec.column}
        if spec.encoding in {"numeric_scaled", "numeric_raw"}:
            components.append(
                html.Div(
                    [
                        html.Label(spec.column),
                        dcc.Input(
                            id=field_id,
                            type="number",
                            value=spec.default_value if spec.default_value is not None else 0,
                            debounce=True,
                        ),
                    ]
                )
            )
        elif spec.encoding in {"one_hot", "ordinal"}:
            categories = spec.categories or ["Unknown"]
            default_value = spec.default_value if spec.default_value is not None else categories[0]
            components.append(
                html.Div(
                    [
                        html.Label(spec.column),
                        dcc.Dropdown(
                            id=field_id,
                            options=[{"label": value, "value": value} for value in categories],
                            value=default_value,
                            clearable=False,
                        ),
                    ]
                )
            )
        else:
            vocabulary = spec.token_vocabulary or []
            components.append(
                html.Div(
                    [
                        html.Label(spec.column),
                        dcc.Dropdown(
                            id=field_id,
                            options=[{"label": token, "value": token} for token in vocabulary],
                            value=[],
                            multi=True,
                            placeholder="Select one or more values",
                        ),
                    ]
                )
            )
    return components


@app.callback(
    Output("prediction-output", "children"),
    Input("estimate-button", "n_clicks"),
    State("model-key", "data"),
    State({"type": "infer-input", "column": ALL}, "value"),
    prevent_initial_call=True,
)
def run_prediction(_n_clicks, model_key, input_values):
    bundle = _cache_get(MODEL_CACHE, model_key)
    if bundle is None:
        return warning_box("Inference blocked", "Train a model first.")

    row: dict[str, object] = {}
    specs = bundle.feature_specs
    values = input_values or []
    for idx, spec in enumerate(specs):
        value = values[idx] if idx < len(values) else None
        if spec.encoding in {"multi_hot_literal", "multi_hot_delimited"}:
            value = value or []
            row[spec.column] = list(value) if isinstance(value, list) else [value]
        else:
            row[spec.column] = value

    frame = make_dynamic_prediction_frame(row)
    try:
        prediction = predict_dynamic(bundle, frame)
    except Exception as exc:
        return warning_box("Inference failed", str(exc))
    return prediction_box(prediction, target_label=bundle.target_column)


if __name__ == "__main__":
    debug_enabled = os.getenv("DASH_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)
