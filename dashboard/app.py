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

# Import supports both `python dashboard/app.py` and package-style imports.
try:
    from .data_model import (
        DEFAULT_MODEL_KEY,
        DynamicModelBundle,
        FeatureEncodingSpec,
        MODEL_OPTIONS,
        default_encoding_for_series,
        infer_column_roles,
        make_dynamic_prediction_frame,
        predict_dynamic,
        train_dynamic_model,
        validate_training_setup,
    )
except ImportError:
    from data_model import (
        DEFAULT_MODEL_KEY,
        DynamicModelBundle,
        FeatureEncodingSpec,
        MODEL_OPTIONS,
        default_encoding_for_series,
        infer_column_roles,
        make_dynamic_prediction_frame,
        predict_dynamic,
        train_dynamic_model,
        validate_training_setup,
    )


# Runtime state is intentionally in memory. The app is a local exploratory tool,
# so uploaded data and fitted models are scoped to the current process.
DATASET_CACHE: dict[str, pd.DataFrame] = {}
MODEL_CACHE: dict[str, DynamicModelBundle] = {}
MAX_CACHE_ITEMS = 8


# Shared UI configuration
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


# Cache and upload helpers
def _cache_put(cache: dict[str, Any], value: Any) -> str:
    """Store a value in a small FIFO cache and return its generated key."""
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
    """Decode Dash's base64 upload payload into a pandas DataFrame."""
    if not contents:
        raise ValueError("Upload payload is empty.")
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    buffer = io.StringIO(decoded.decode("utf-8-sig"))
    return pd.read_csv(buffer)


def _pick_default_target(df: pd.DataFrame, roles: dict[str, str]) -> str:
    """Choose a useful initial target, favoring outcome/CIRV-style numeric columns."""
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


# Formatting and reusable UI components
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


def prediction_box(
    prediction,
    target_label: str = "Target",
    model_label: str | None = None,
) -> html.Div:
    interval_caption = "90% model interval"
    if model_label and "forest" in model_label.lower():
        interval_caption = "Tree ensemble interval (p5-p95)"
    return html.Div(
        [
            html.Span(f"Estimated {target_label}", className="prediction-label"),
            html.Strong(format_target_value(prediction.prediction)),
            html.P(
                f"{interval_caption}: {format_target_value(prediction.lower)}"
                f" to {format_target_value(prediction.upper)}"
            ),
            html.Small(f"Predictive standard deviation: {format_target_value(prediction.std)}"),
            html.Small(f"Model: {model_label}" if model_label else ""),
        ],
        className="prediction-box",
    )


# Model quality messaging
def pitfalls_box(messages: list[str], stale: bool = False) -> html.Div:
    class_name = "pitfalls-box"
    if stale:
        class_name += " stale"
    return html.Div(
        [
            html.H3("Pitfalls"),
            html.Ul([html.Li(message) for message in messages]),
        ],
        className=class_name,
    )


def _same_feature_configuration(bundle: DynamicModelBundle, current_specs: list[FeatureEncodingSpec]) -> bool:
    trained = sorted(
        [
            (
                spec.column,
                spec.encoding,
                spec.delimiter if spec.encoding == "multi_hot_delimited" else None,
            )
            for spec in bundle.feature_specs
        ]
    )
    current = sorted(
        [
            (
                spec.column,
                spec.encoding,
                spec.delimiter if spec.encoding == "multi_hot_delimited" else None,
            )
            for spec in current_specs
        ]
    )
    return trained == current


def _selection_is_stale(
    bundle: DynamicModelBundle,
    selected_model: str | None,
    selected_target: str | None,
    selected_specs: list[FeatureEncodingSpec],
) -> bool:
    if selected_model and selected_model != bundle.model_key:
        return True
    if selected_target and selected_target != bundle.target_column:
        return True
    return not _same_feature_configuration(bundle, selected_specs)


def _latest_r2_message(bundle: DynamicModelBundle) -> str:
    test_r2 = float(bundle.metrics.get("test_r2", float("nan")))
    if not math.isfinite(test_r2):
        return "Last trained model has no reliable holdout R2 yet."
    return f"Last trained model test R2: {test_r2:.3f}."


def _pitfall_intro(bundle: DynamicModelBundle) -> str:
    return f"Warnings are based on the currently trained {bundle.model_label} model."


def _empty_pitfall_message() -> list[str]:
    return ["Train a model to see adaptive warnings and quality checks."]


def _training_error_response(
    title: str,
    message: str,
    figure_message: str,
    pitfall_messages: list[str] | None = None,
):
    figures = [empty_figure(figure_message) for _ in range(5)]
    return (
        None,
        warning_box(title, message),
        [],
        *figures,
        pitfalls_box(pitfall_messages or _empty_pitfall_message()),
    )


def _stale_pitfall_messages(bundle: DynamicModelBundle) -> list[str]:
    return [
        "Selections changed after training. Retrain to get warnings for the current setup.",
        _latest_r2_message(bundle),
    ]


def _pitfall_messages(bundle: DynamicModelBundle) -> list[str]:
    messages: list[str] = []
    metrics = bundle.metrics
    test_r2 = float(metrics.get("test_r2", float("nan")))
    train_r2 = float(metrics.get("train_r2", float("nan")))
    test_rows = int(metrics.get("test_rows", 0.0))
    train_rows = int(metrics.get("train_rows", 0.0))

    if not math.isfinite(test_r2) or test_rows <= 0:
        messages.append("Not enough holdout data to judge model quality. Train with more rows for a reliable check.")
        return messages

    if test_r2 >= 0.85:
        messages.append("R2 is very high. Be careful, this model might be overfitting.")
    elif test_r2 < 0.0:
        messages.append("R2 is below 0. This model performs worse than a simple average baseline.")
    elif test_r2 < 0.1:
        messages.append("R2 is near 0. This model has no significant predictive performance.")

    if math.isfinite(train_r2) and (train_r2 - test_r2) > 0.25 and train_r2 > 0.6:
        messages.append("Training performance is much better than holdout performance. The model may be overfitting.")

    if test_rows < 30:
        messages.append("The holdout sample is small. Evaluation metrics may move a lot with new data.")

    encoded_features = len(bundle.encoded_feature_names)
    total_rows = max(train_rows + test_rows, 1)
    if encoded_features > max(100, int(0.5 * total_rows)):
        messages.append("You have many encoded features for the available rows. Predictions may be unstable.")

    if "forest" in bundle.model_label.lower() and total_rows < 200:
        messages.append("Random Forest on small datasets can vary across samples. Validate with more data if possible.")

    if not messages:
        messages.append("No major warning signs right now, but treat predictions as directional and validate on fresh data.")
    return messages


# Chart helpers
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


# App layout
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
                        html.H1("Not a dashboard"),
                        html.P("The decision assistant.", className="lede"),
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
                                html.Div(
                                    [
                                        html.Label("Model type"),
                                        dcc.Dropdown(
                                            id="model-type",
                                            options=MODEL_OPTIONS,
                                            value=DEFAULT_MODEL_KEY,
                                            clearable=False,
                                        ),
                                    ]
                                ),
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
                        html.Div(id="pitfalls-box"),
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


# Callbacks
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
    Output("pitfalls-box", "children", allow_duplicate=True),
    Input("model-key", "data"),
    Input("model-type", "value"),
    Input("target-column", "value"),
    Input("treatment-columns", "value"),
    Input({"type": "encoding-select", "column": ALL}, "value"),
    Input({"type": "encoding-delimiter", "column": ALL}, "value"),
    prevent_initial_call=True,
)
def refresh_pitfalls_from_selection(
    model_key,
    selected_model,
    selected_target,
    selected_treatments,
    selected_encodings,
    selected_delimiters,
):
    bundle = _cache_get(MODEL_CACHE, model_key)
    if bundle is None:
        return pitfalls_box(_empty_pitfall_message())

    specs = _build_feature_specs(
        selected_treatments or [],
        selected_encodings or [],
        selected_delimiters or [],
    )
    stale = _selection_is_stale(bundle, selected_model, selected_target, specs)
    if stale:
        return pitfalls_box(_stale_pitfall_messages(bundle), stale=True)
    return pitfalls_box([_pitfall_intro(bundle), *_pitfall_messages(bundle)])


@app.callback(
    Output("model-key", "data"),
    Output("train-status", "children"),
    Output("model-metrics", "children"),
    Output("target-distribution", "figure"),
    Output("feature-target-corr", "figure"),
    Output("feature-corr-heatmap", "figure"),
    Output("holdout-prediction", "figure"),
    Output("holdout-residual", "figure"),
    Output("pitfalls-box", "children"),
    Input("train-button", "n_clicks"),
    State("dataset-key", "data"),
    State("model-type", "value"),
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
    model_type,
    target_column,
    treatments,
    encoding_values,
    delimiter_values,
    corr_method,
    corr_top_n,
):
    df = _cache_get(DATASET_CACHE, dataset_key)
    if df is None:
        return _training_error_response(
            "Training blocked",
            "Upload a CSV first.",
            "Upload data before training",
        )

    errors = validate_training_setup(df, target_column, treatments)
    if errors:
        return _training_error_response(
            "Training blocked",
            " ".join(errors),
            "Fix setup errors before training",
        )

    specs = _build_feature_specs(treatments or [], encoding_values or [], delimiter_values or [])
    try:
        bundle = train_dynamic_model(
            df,
            str(target_column),
            specs,
            model_key=str(model_type or DEFAULT_MODEL_KEY),
        )
    except Exception as exc:
        return _training_error_response(
            "Training failed",
            str(exc),
            "Training failed",
            ["Training failed, so model quality warnings are not available yet."],
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
            html.P(
                f"Training complete with {bundle.model_label}. You can inspect diagnostics and run inference below.",
                className="muted",
            ),
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
        pitfalls_box([_pitfall_intro(bundle), *_pitfall_messages(bundle)]),
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
    return prediction_box(
        prediction,
        target_label=bundle.target_column,
        model_label=bundle.model_label,
    )


if __name__ == "__main__":
    debug_enabled = os.getenv("DASH_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)
