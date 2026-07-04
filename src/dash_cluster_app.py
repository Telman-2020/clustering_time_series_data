from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    dash_table,
    dcc,
    html,
    no_update,
)
from plotly.subplots import make_subplots

from cluster_time_series_methods import (
    DEFAULT_HDBSCAN_METRIC,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_HDBSCAN_SELECTION_EPSILON,
    DEFAULT_HDBSCAN_SELECTION_METHOD,
    DEFAULT_HDBSCAN_TOTAL_LABELS,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_KMEANS_MAX_ITER,
    DEFAULT_KMEANS_OUTLIER_QUANTILE,
    DEFAULT_KMEANS_RANDOM_STATE,
    DEFAULT_KMEANS_TOLERANCE,
    prepare_selected_feature_matrix,
    run_hdbscan,
    run_kmeans,
)
from prepared_data import (
    DEFAULT_CLUSTERING_FEATURES,
    get_available_numeric_features,
    load_prepared_data,
    validate_default_features,
)


# ============================================================
# Method names
# ============================================================

KMEANS_METHOD = "KMeans"
HDBSCAN_METHOD = "HDBSCAN"


# ============================================================
# Load prepared Parquet data
# ============================================================

print("Loading prepared Parquet dataset...")

PREPARED_DATA = load_prepared_data()
validate_default_features(PREPARED_DATA)

AVAILABLE_FEATURES = get_available_numeric_features(
    PREPARED_DATA
)

DEFAULT_MODEL_FEATURES = [
    feature
    for feature in DEFAULT_CLUSTERING_FEATURES
    if feature in AVAILABLE_FEATURES
]

DEFAULT_PLOT_FEATURES = [
    feature
    for feature in [
        "gx",
        "gy",
        "gz",
        "bx",
        "by",
        "bz",
        "pipe_rotation",
        "flow_rotation_lower",
        "flow_rotation_upper",
    ]
    if feature in AVAILABLE_FEATURES
]

if not DEFAULT_PLOT_FEATURES:
    DEFAULT_PLOT_FEATURES = AVAILABLE_FEATURES[:6]


# ============================================================
# In-memory model cache
# ============================================================

DATASETS: dict[str, pd.DataFrame] = {}
LAST_RUN_METADATA: dict[str, Any] = {}


# ============================================================
# Model execution
# ============================================================

def run_both_algorithms(
    selected_features: list[str],
    *,
    kmeans_clusters: int,
    kmeans_max_iter: int,
    kmeans_tolerance: float,
    kmeans_random_state: int,
    kmeans_outlier_quantile: float,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_metric: str,
    hdbscan_selection_method: str,
    hdbscan_epsilon: float,
    hdbscan_total_labels: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Run KMeans and HDBSCAN using the same eligible rows,
    selected features and scaled feature matrix.
    """
    started_at = time.perf_counter()

    (
        eligible_data,
        feature_matrix,
        preparation_metadata,
    ) = prepare_selected_feature_matrix(
        data=PREPARED_DATA,
        selected_features=selected_features,
    )

    kmeans_data, kmeans_metadata = run_kmeans(
        data=eligible_data,
        feature_matrix=feature_matrix,
        n_clusters=int(kmeans_clusters),
        max_iter=int(kmeans_max_iter),
        tolerance=float(kmeans_tolerance),
        random_state=int(kmeans_random_state),
        outlier_quantile=float(
            kmeans_outlier_quantile
        ),
    )

    hdbscan_data, hdbscan_metadata = run_hdbscan(
        data=eligible_data,
        feature_matrix=feature_matrix,
        min_cluster_size=int(
            hdbscan_min_cluster_size
        ),
        min_samples=int(hdbscan_min_samples),
        metric=str(hdbscan_metric),
        cluster_selection_method=str(
            hdbscan_selection_method
        ),
        cluster_selection_epsilon=float(
            hdbscan_epsilon
        ),
        target_total_labels=int(
            hdbscan_total_labels
        ),
    )

    elapsed_seconds = (
        time.perf_counter() - started_at
    )

    metadata = {
        **preparation_metadata,
        "elapsed_seconds": elapsed_seconds,
        "kmeans": kmeans_metadata,
        "hdbscan": hdbscan_metadata,
    }

    return {
        KMEANS_METHOD: kmeans_data,
        HDBSCAN_METHOD: hdbscan_data,
    }, metadata


print("Running initial models...")

DATASETS, LAST_RUN_METADATA = run_both_algorithms(
    DEFAULT_MODEL_FEATURES,
    kmeans_clusters=DEFAULT_KMEANS_CLUSTERS,
    kmeans_max_iter=DEFAULT_KMEANS_MAX_ITER,
    kmeans_tolerance=DEFAULT_KMEANS_TOLERANCE,
    kmeans_random_state=(
        DEFAULT_KMEANS_RANDOM_STATE
    ),
    kmeans_outlier_quantile=(
        DEFAULT_KMEANS_OUTLIER_QUANTILE
    ),
    hdbscan_min_cluster_size=(
        DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
    ),
    hdbscan_min_samples=(
        DEFAULT_HDBSCAN_MIN_SAMPLES
    ),
    hdbscan_metric=DEFAULT_HDBSCAN_METRIC,
    hdbscan_selection_method=(
        DEFAULT_HDBSCAN_SELECTION_METHOD
    ),
    hdbscan_epsilon=(
        DEFAULT_HDBSCAN_SELECTION_EPSILON
    ),
    hdbscan_total_labels=(
        DEFAULT_HDBSCAN_TOTAL_LABELS
    ),
)

print("Initial models completed.")


# ============================================================
# Feature helpers
# ============================================================

def feature_coverage(feature: str) -> float:
    return (
        100.0
        * PREPARED_DATA[feature].notna().mean()
    )


def feature_options() -> list[dict[str, str]]:
    return [
        {
            "label": (
                f"{feature} "
                f"({feature_coverage(feature):.2f}% coverage)"
            ),
            "value": feature,
        }
        for feature in AVAILABLE_FEATURES
    ]


# ============================================================
# Data helpers
# ============================================================

def get_dataset(method: str) -> pd.DataFrame:
    return DATASETS[method].copy()


def filter_trip(
    data: pd.DataFrame,
    selected_trip: Any,
) -> pd.DataFrame:
    if selected_trip == "all":
        return data.copy()

    return data[
        data["trip_id"] == int(selected_trip)
    ].copy()


def trip_options() -> list[dict[str, Any]]:
    trips = sorted(
        int(value)
        for value in PREPARED_DATA[
            "trip_id"
        ].dropna().unique()
    )

    return [
        {
            "label": "All trips",
            "value": "all",
        }
    ] + [
        {
            "label": f"Trip {trip}",
            "value": trip,
        }
        for trip in trips
    ]


def x_axis_options() -> list[dict[str, str]]:
    candidates = [
        (
            "row_in_trip",
            "Row number inside trip",
        ),
        (
            "depth",
            "Depth",
        ),
        (
            "time",
            "Time",
        ),
        (
            "ground_depth",
            "Ground depth",
        ),
        (
            "ground_time",
            "Ground time",
        ),
    ]

    return [
        {
            "label": label,
            "value": column,
        }
        for column, label in candidates
        if column in PREPARED_DATA.columns
    ]


def default_x_axis() -> str:
    available = [
        option["value"]
        for option in x_axis_options()
    ]

    if "depth" in available:
        return "depth"

    return "row_in_trip"


def cluster_label(
    method: str,
    cluster_value: int,
) -> str:
    if (
        method == HDBSCAN_METHOD
        and cluster_value == -1
    ):
        return "Anomaly / Noise"

    return f"Cluster {cluster_value}"


def cluster_color_mapping(
    method: str,
    cluster_values: list[int],
) -> dict[int, str]:
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
    )

    mapping: dict[int, str] = {}

    normal_clusters = [
        cluster
        for cluster in cluster_values
        if cluster != -1
    ]

    if (
        method == HDBSCAN_METHOD
        and -1 in cluster_values
    ):
        mapping[-1] = "black"

    for index, cluster in enumerate(
        normal_clusters
    ):
        mapping[cluster] = palette[
            index % len(palette)
        ]

    return mapping


def cluster_dropdown_options(
    method: str,
    selected_trip: Any,
) -> tuple[
    list[dict[str, Any]],
    list[int],
]:
    data = filter_trip(
        get_dataset(method),
        selected_trip,
    )

    cluster_values = sorted(
        int(value)
        for value in data[
            "cluster"
        ].dropna().unique()
    )

    options = [
        {
            "label": cluster_label(
                method,
                cluster,
            ),
            "value": cluster,
        }
        for cluster in cluster_values
    ]

    return options, cluster_values


# ============================================================
# Summary tables
# ============================================================

def create_filtered_summary(
    data: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
        )
        .reset_index()
    )

    summary["cluster_label"] = summary[
        "cluster"
    ].apply(
        lambda value: cluster_label(
            method,
            int(value),
        )
    )

    summary["percent_of_filtered_data"] = (
        100.0
        * summary["rows"]
        / len(data)
    )

    if method == KMEANS_METHOD:
        if "possible_outlier" in data.columns:
            outliers = (
                data.groupby("cluster")[
                    "possible_outlier"
                ]
                .sum()
                .reset_index()
                .rename(
                    columns={
                        "possible_outlier":
                            "possible_outliers"
                    }
                )
            )

            summary = summary.merge(
                outliers,
                on="cluster",
                how="left",
            )

        if "cluster_distance" in data.columns:
            distances = (
                data.groupby("cluster")[
                    "cluster_distance"
                ]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "cluster_distance":
                            "average_cluster_distance"
                    }
                )
            )

            summary = summary.merge(
                distances,
                on="cluster",
                how="left",
            )

    if method == HDBSCAN_METHOD:
        summary["cluster_type"] = summary[
            "cluster"
        ].apply(
            lambda value: (
                "anomaly_noise"
                if int(value) == -1
                else "behavioural_regime"
            )
        )

        if (
            "cluster_probability"
            in data.columns
        ):
            probabilities = (
                data.groupby("cluster")[
                    "cluster_probability"
                ]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "cluster_probability":
                            "average_membership_probability"
                    }
                )
            )

            summary = summary.merge(
                probabilities,
                on="cluster",
                how="left",
            )

            summary.loc[
                summary["cluster"] == -1,
                "average_membership_probability",
            ] = np.nan

    front_columns = [
        "cluster",
        "cluster_label",
        "rows",
        "percent_of_filtered_data",
    ]

    remaining_columns = [
        column
        for column in summary.columns
        if column not in front_columns
    ]

    return (
        summary[
            front_columns
            + remaining_columns
        ]
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def sanitize_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    for column in result.columns:
        if pd.api.types.is_float_dtype(
            result[column]
        ):
            result[column] = result[
                column
            ].round(4)

        result[column] = result[
            column
        ].apply(
            lambda value: (
                None
                if pd.isna(value)
                else value
            )
        )

    return result


# ============================================================
# Plot helpers
# ============================================================

def downsample_rows(
    data: pd.DataFrame,
    maximum_points: int,
) -> pd.DataFrame:
    if (
        maximum_points <= 0
        or len(data) <= maximum_points
    ):
        return data

    positions = np.linspace(
        0,
        len(data) - 1,
        maximum_points,
    ).astype(int)

    return data.iloc[positions].copy()


def create_method_figure(
    data: pd.DataFrame,
    method: str,
    selected_features: list[str],
    x_axis: str,
    selected_clusters: list[int],
    maximum_points: int,
) -> go.Figure:
    valid_features = [
        feature
        for feature in selected_features
        if feature in data.columns
    ]

    if not valid_features:
        figure = go.Figure()
        figure.update_layout(
            title=(
                "Select at least one feature "
                "to plot."
            )
        )
        return figure

    filtered = data[
        data["cluster"].isin(
            selected_clusters
        )
    ].copy()

    if filtered.empty:
        figure = go.Figure()
        figure.update_layout(
            title="No data for selected filters."
        )
        return figure

    if x_axis not in filtered.columns:
        x_axis = "row_in_trip"

    filtered = filtered.sort_values(
        ["trip_id", x_axis],
        kind="stable",
    )

    filtered = downsample_rows(
        filtered,
        maximum_points,
    )

    cluster_values = sorted(
        int(value)
        for value in filtered[
            "cluster"
        ].dropna().unique()
    )

    color_mapping = cluster_color_mapping(
        method,
        cluster_values,
    )

    figure = make_subplots(
        rows=len(valid_features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=valid_features,
    )

    for row_index, feature in enumerate(
        valid_features,
        start=1,
    ):
        for cluster in cluster_values:
            cluster_data = filtered[
                filtered["cluster"] == cluster
            ]

            display_label = cluster_label(
                method,
                cluster,
            )

            figure.add_trace(
                go.Scattergl(
                    x=cluster_data[x_axis],
                    y=cluster_data[feature],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": color_mapping[
                            cluster
                        ],
                        "opacity": 0.75,
                    },
                    name=display_label,
                    legendgroup=str(cluster),
                    showlegend=(row_index == 1),
                    hovertemplate=(
                        f"Feature: {feature}"
                        "<br>Value: %{y}"
                        "<br>X: %{x}"
                        f"<br>{display_label}"
                        "<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )

        figure.update_yaxes(
            title_text=feature,
            row=row_index,
            col=1,
        )

    figure.update_xaxes(
        title_text=x_axis,
        row=len(valid_features),
        col=1,
    )

    figure.update_layout(
        title=(
            f"{method} behavioural "
            "segmentation"
        ),
        height=max(
            420,
            260 * len(valid_features),
        ),
        template="plotly_white",
        legend_title="Regime",
        margin={
            "l": 60,
            "r": 20,
            "t": 80,
            "b": 50,
        },
    )

    return figure


# ============================================================
# Status helpers
# ============================================================

def initial_status_text() -> str:
    hdbscan_metadata = LAST_RUN_METADATA[
        "hdbscan"
    ]

    return (
        f"Initial models completed using "
        f"{len(DEFAULT_MODEL_FEATURES)} features. "
        f"Eligible rows: "
        f"{LAST_RUN_METADATA['eligible_rows']:,}. "
        f"HDBSCAN Anomaly / Noise: "
        f"{hdbscan_metadata['noise_rows']:,} rows "
        f"({hdbscan_metadata['noise_percent']:.2f}%)."
    )


def numeric_control(
    label: str,
    component_id: str,
    value: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> html.Div:
    return html.Div(
        children=[
            html.Label(
                label,
                style={
                    "fontWeight": "bold",
                },
            ),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                debounce=True,
                style={
                    "width": "100%",
                    "height": "36px",
                },
            ),
        ]
    )


# ============================================================
# Dash layout
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "margin": "20px auto",
        "maxWidth": "1900px",
    },
    children=[
        dcc.Store(
            id="model-version",
            data=0,
        ),
        dcc.Store(
            id="applied-features-store",
            data=DEFAULT_MODEL_FEATURES,
        ),

        html.H1(
            "KMeans vs HDBSCAN "
            "Behaviour Segmentation"
        ),

        html.P(
            "The dashboard reads the prepared "
            "Parquet feature dataset and reruns "
            "both clustering methods using the "
            "same selected rows and scaled features."
        ),

        html.Div(
            style={
                "padding": "14px",
                "border": "1px solid #d9d9d9",
                "borderRadius": "8px",
                "marginBottom": "18px",
            },
            children=[
                html.Label(
                    "Features used by both algorithms",
                    style={
                        "fontWeight": "bold",
                    },
                ),

                dcc.Dropdown(
                    id="model-feature-dropdown",
                    options=feature_options(),
                    value=DEFAULT_MODEL_FEATURES,
                    multi=True,
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns":
                            "1fr 1fr",
                        "gap": "18px",
                        "marginTop": "16px",
                    },
                    children=[
                        html.Div(
                            style={
                                "padding": "12px",
                                "border":
                                    "1px solid #dedede",
                                "borderRadius": "8px",
                            },
                            children=[
                                html.H3(
                                    "KMeans parameters"
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns":
                                            "repeat(2, 1fr)",
                                        "gap": "12px",
                                    },
                                    children=[
                                        numeric_control(
                                            "Clusters",
                                            "kmeans-clusters",
                                            DEFAULT_KMEANS_CLUSTERS,
                                            minimum=2,
                                            maximum=50,
                                            step=1,
                                        ),
                                        numeric_control(
                                            "Maximum iterations",
                                            "kmeans-max-iter",
                                            DEFAULT_KMEANS_MAX_ITER,
                                            minimum=1,
                                            maximum=2000,
                                            step=10,
                                        ),
                                        numeric_control(
                                            "Tolerance",
                                            "kmeans-tolerance",
                                            DEFAULT_KMEANS_TOLERANCE,
                                            minimum=1e-8,
                                            maximum=0.1,
                                            step=0.0001,
                                        ),
                                        numeric_control(
                                            "Random state",
                                            "kmeans-random-state",
                                            DEFAULT_KMEANS_RANDOM_STATE,
                                            minimum=0,
                                            maximum=1000000,
                                            step=1,
                                        ),
                                        numeric_control(
                                            "Outlier quantile",
                                            "kmeans-outlier-quantile",
                                            DEFAULT_KMEANS_OUTLIER_QUANTILE,
                                            minimum=0.5,
                                            maximum=0.9999,
                                            step=0.001,
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            style={
                                "padding": "12px",
                                "border":
                                    "1px solid #dedede",
                                "borderRadius": "8px",
                            },
                            children=[
                                html.H3(
                                    "HDBSCAN parameters"
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns":
                                            "repeat(2, 1fr)",
                                        "gap": "12px",
                                    },
                                    children=[
                                        numeric_control(
                                            "Minimum cluster size",
                                            "hdbscan-min-cluster-size",
                                            DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                                            minimum=2,
                                            maximum=100000,
                                            step=10,
                                        ),
                                        numeric_control(
                                            "Minimum samples",
                                            "hdbscan-min-samples",
                                            DEFAULT_HDBSCAN_MIN_SAMPLES,
                                            minimum=1,
                                            maximum=100000,
                                            step=5,
                                        ),

                                        html.Div(
                                            children=[
                                                html.Label(
                                                    "Distance metric",
                                                    style={
                                                        "fontWeight":
                                                            "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="hdbscan-metric",
                                                    options=[
                                                        {
                                                            "label":
                                                                "Euclidean",
                                                            "value":
                                                                "euclidean",
                                                        },
                                                        {
                                                            "label":
                                                                "Manhattan",
                                                            "value":
                                                                "manhattan",
                                                        },
                                                        {
                                                            "label":
                                                                "Chebyshev",
                                                            "value":
                                                                "chebyshev",
                                                        },
                                                    ],
                                                    value=(
                                                        DEFAULT_HDBSCAN_METRIC
                                                    ),
                                                    clearable=False,
                                                ),
                                            ]
                                        ),

                                        html.Div(
                                            children=[
                                                html.Label(
                                                    "Selection method",
                                                    style={
                                                        "fontWeight":
                                                            "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="hdbscan-selection-method",
                                                    options=[
                                                        {
                                                            "label":
                                                                "EOM",
                                                            "value":
                                                                "eom",
                                                        },
                                                        {
                                                            "label":
                                                                "Leaf",
                                                            "value":
                                                                "leaf",
                                                        },
                                                    ],
                                                    value=(
                                                        DEFAULT_HDBSCAN_SELECTION_METHOD
                                                    ),
                                                    clearable=False,
                                                ),
                                            ]
                                        ),

                                        numeric_control(
                                            "Selection epsilon",
                                            "hdbscan-epsilon",
                                            DEFAULT_HDBSCAN_SELECTION_EPSILON,
                                            minimum=0.0,
                                            maximum=1000.0,
                                            step=0.01,
                                        ),

                                        numeric_control(
                                            "Total displayed labels",
                                            "hdbscan-total-labels",
                                            DEFAULT_HDBSCAN_TOTAL_LABELS,
                                            minimum=2,
                                            maximum=50,
                                            step=1,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                html.Button(
                    "Apply settings and rerun models",
                    id="run-models-button",
                    n_clicks=0,
                    style={
                        "marginTop": "14px",
                        "height": "40px",
                        "fontWeight": "bold",
                    },
                ),

                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="model-status",
                        children=initial_status_text(),
                        style={
                            "marginTop": "12px",
                            "fontWeight": "bold",
                        },
                    ),
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "1fr 1fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Trip"),
                        dcc.Dropdown(
                            id="trip-dropdown",
                            options=trip_options(),
                            value="all",
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label("X-axis"),
                        dcc.Dropdown(
                            id="x-axis-dropdown",
                            options=x_axis_options(),
                            value=default_x_axis(),
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label(
                            "Maximum points per chart"
                        ),
                        dcc.Input(
                            id="max-points-input",
                            type="number",
                            value=12000,
                            min=1000,
                            step=1000,
                            style={
                                "width": "100%",
                            },
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "1fr 1fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label(
                            "Features to plot"
                        ),
                        dcc.Dropdown(
                            id="plot-feature-dropdown",
                            options=feature_options(),
                            value=DEFAULT_PLOT_FEATURES,
                            multi=True,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label(
                            "KMeans clusters"
                        ),
                        dcc.Dropdown(
                            id="kmeans-cluster-dropdown",
                            multi=True,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label(
                            "HDBSCAN labels"
                        ),
                        dcc.Dropdown(
                            id="hdbscan-cluster-dropdown",
                            multi=True,
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "1fr 1fr",
                "gap": "16px",
                "alignItems": "start",
            },
            children=[
                dcc.Graph(
                    id="kmeans-graph"
                ),
                dcc.Graph(
                    id="hdbscan-graph"
                ),
            ],
        ),

        html.H2("Cluster summaries"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "1fr 1fr",
                "gap": "16px",
            },
            children=[
                html.Div(
                    children=[
                        html.H3(
                            "KMeans summary"
                        ),
                        dash_table.DataTable(
                            id="kmeans-summary-table",
                            page_size=20,
                            sort_action="native",
                            filter_action="native",
                            style_table={
                                "overflowX": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "6px",
                                "fontSize": "13px",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor":
                                    "#f2f2f2",
                            },
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.H3(
                            "HDBSCAN summary"
                        ),
                        dash_table.DataTable(
                            id="hdbscan-summary-table",
                            page_size=20,
                            sort_action="native",
                            filter_action="native",
                            style_table={
                                "overflowX": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "6px",
                                "fontSize": "13px",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor":
                                    "#f2f2f2",
                            },
                            style_data_conditional=cast(
                                Any,
                                [
                                    {
                                        "if": {
                                            "filter_query":
                                                "{cluster} = -1",
                                        },
                                        "backgroundColor":
                                            "black",
                                        "color": "white",
                                        "fontWeight": "bold",
                                    }
                                ],
                            ),
                        ),
                    ]
                ),
            ],
        ),
    ],
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("model-version", "data"),
    Output("applied-features-store", "data"),
    Output("model-status", "children"),
    Input("run-models-button", "n_clicks"),
    State("model-feature-dropdown", "value"),
    State("kmeans-clusters", "value"),
    State("kmeans-max-iter", "value"),
    State("kmeans-tolerance", "value"),
    State("kmeans-random-state", "value"),
    State("kmeans-outlier-quantile", "value"),
    State("hdbscan-min-cluster-size", "value"),
    State("hdbscan-min-samples", "value"),
    State("hdbscan-metric", "value"),
    State("hdbscan-selection-method", "value"),
    State("hdbscan-epsilon", "value"),
    State("hdbscan-total-labels", "value"),
    prevent_initial_call=True,
)
def rerun_models(
    n_clicks: int,
    selected_features: list[str] | None,
    kmeans_clusters: int | None,
    kmeans_max_iter: int | None,
    kmeans_tolerance: float | None,
    kmeans_random_state: int | None,
    kmeans_outlier_quantile: float | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    hdbscan_metric: str | None,
    hdbscan_selection_method: str | None,
    hdbscan_epsilon: float | None,
    hdbscan_total_labels: int | None,
):
    del n_clicks

    if not selected_features:
        return (
            no_update,
            no_update,
            "Select at least one model feature.",
        )

    try:
        datasets, metadata = run_both_algorithms(
            selected_features,
            kmeans_clusters=int(
                kmeans_clusters
                or DEFAULT_KMEANS_CLUSTERS
            ),
            kmeans_max_iter=int(
                kmeans_max_iter
                or DEFAULT_KMEANS_MAX_ITER
            ),
            kmeans_tolerance=float(
                kmeans_tolerance
                if kmeans_tolerance is not None
                else DEFAULT_KMEANS_TOLERANCE
            ),
            kmeans_random_state=int(
                kmeans_random_state
                if kmeans_random_state is not None
                else DEFAULT_KMEANS_RANDOM_STATE
            ),
            kmeans_outlier_quantile=float(
                kmeans_outlier_quantile
                if kmeans_outlier_quantile is not None
                else DEFAULT_KMEANS_OUTLIER_QUANTILE
            ),
            hdbscan_min_cluster_size=int(
                hdbscan_min_cluster_size
                or DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
            ),
            hdbscan_min_samples=int(
                hdbscan_min_samples
                or DEFAULT_HDBSCAN_MIN_SAMPLES
            ),
            hdbscan_metric=(
                hdbscan_metric
                or DEFAULT_HDBSCAN_METRIC
            ),
            hdbscan_selection_method=(
                hdbscan_selection_method
                or DEFAULT_HDBSCAN_SELECTION_METHOD
            ),
            hdbscan_epsilon=float(
                hdbscan_epsilon
                if hdbscan_epsilon is not None
                else DEFAULT_HDBSCAN_SELECTION_EPSILON
            ),
            hdbscan_total_labels=int(
                hdbscan_total_labels
                or DEFAULT_HDBSCAN_TOTAL_LABELS
            ),
        )

        DATASETS.clear()
        DATASETS.update(datasets)

        LAST_RUN_METADATA.clear()
        LAST_RUN_METADATA.update(metadata)

        hdbscan_metadata = metadata["hdbscan"]
        kmeans_metadata = metadata["kmeans"]

        status = (
            f"Models completed in "
            f"{metadata['elapsed_seconds']:.2f} seconds. "
            f"Used {metadata['eligible_rows']:,} of "
            f"{metadata['raw_rows']:,} rows. "
            f"KMeans: "
            f"{kmeans_metadata['clusters']} clusters. "
            f"HDBSCAN: "
            f"{hdbscan_metadata['displayed_normal_clusters']} "
            f"regimes plus Anomaly / Noise "
            f"({hdbscan_metadata['noise_rows']:,} rows; "
            f"{hdbscan_metadata['noise_percent']:.2f}%)."
        )

        return (
            int(time.time() * 1000),
            selected_features,
            status,
        )

    except Exception as error:
        return (
            no_update,
            no_update,
            f"Model run failed: {error}",
        )


@app.callback(
    Output(
        "kmeans-cluster-dropdown",
        "options",
    ),
    Output(
        "kmeans-cluster-dropdown",
        "value",
    ),
    Output(
        "hdbscan-cluster-dropdown",
        "options",
    ),
    Output(
        "hdbscan-cluster-dropdown",
        "value",
    ),
    Input("model-version", "data"),
    Input("trip-dropdown", "value"),
)
def update_cluster_dropdowns(
    model_version: int,
    selected_trip: Any,
):
    del model_version

    (
        kmeans_options,
        kmeans_values,
    ) = cluster_dropdown_options(
        KMEANS_METHOD,
        selected_trip,
    )

    (
        hdbscan_options,
        hdbscan_values,
    ) = cluster_dropdown_options(
        HDBSCAN_METHOD,
        selected_trip,
    )

    return (
        kmeans_options,
        kmeans_values,
        hdbscan_options,
        hdbscan_values,
    )


@app.callback(
    Output("kmeans-graph", "figure"),
    Output("hdbscan-graph", "figure"),
    Output("kmeans-summary-table", "data"),
    Output("kmeans-summary-table", "columns"),
    Output("hdbscan-summary-table", "data"),
    Output("hdbscan-summary-table", "columns"),
    Input("model-version", "data"),
    Input("trip-dropdown", "value"),
    Input("x-axis-dropdown", "value"),
    Input("plot-feature-dropdown", "value"),
    Input("kmeans-cluster-dropdown", "value"),
    Input("hdbscan-cluster-dropdown", "value"),
    Input("max-points-input", "value"),
)
def update_dashboard(
    model_version: int,
    selected_trip: Any,
    x_axis: str,
    selected_plot_features: list[str] | None,
    selected_kmeans_clusters: list[int] | None,
    selected_hdbscan_clusters: list[int] | None,
    maximum_points: int | None,
):
    del model_version

    kmeans_data = filter_trip(
        get_dataset(KMEANS_METHOD),
        selected_trip,
    )

    hdbscan_data = filter_trip(
        get_dataset(HDBSCAN_METHOD),
        selected_trip,
    )

    if not selected_kmeans_clusters:
        selected_kmeans_clusters = sorted(
            int(value)
            for value in kmeans_data[
                "cluster"
            ].dropna().unique()
        )

    if not selected_hdbscan_clusters:
        selected_hdbscan_clusters = sorted(
            int(value)
            for value in hdbscan_data[
                "cluster"
            ].dropna().unique()
        )

    plot_features = (
        selected_plot_features or []
    )

    max_points = int(
        maximum_points or 12000
    )

    kmeans_figure = create_method_figure(
        data=kmeans_data,
        method=KMEANS_METHOD,
        selected_features=plot_features,
        x_axis=x_axis,
        selected_clusters=[
            int(value)
            for value in selected_kmeans_clusters
        ],
        maximum_points=max_points,
    )

    hdbscan_figure = create_method_figure(
        data=hdbscan_data,
        method=HDBSCAN_METHOD,
        selected_features=plot_features,
        x_axis=x_axis,
        selected_clusters=[
            int(value)
            for value in selected_hdbscan_clusters
        ],
        maximum_points=max_points,
    )

    filtered_kmeans = kmeans_data[
        kmeans_data["cluster"].isin(
            selected_kmeans_clusters
        )
    ]

    filtered_hdbscan = hdbscan_data[
        hdbscan_data["cluster"].isin(
            selected_hdbscan_clusters
        )
    ]

    kmeans_summary = sanitize_table(
        create_filtered_summary(
            filtered_kmeans,
            KMEANS_METHOD,
        )
    )

    hdbscan_summary = sanitize_table(
        create_filtered_summary(
            filtered_hdbscan,
            HDBSCAN_METHOD,
        )
    )

    kmeans_columns = [
        {
            "name": column,
            "id": column,
        }
        for column in kmeans_summary.columns
    ]

    hdbscan_columns = [
        {
            "name": column,
            "id": column,
        }
        for column in hdbscan_summary.columns
    ]

    return (
        kmeans_figure,
        hdbscan_figure,
        kmeans_summary.to_dict("records"),
        kmeans_columns,
        hdbscan_summary.to_dict("records"),
        hdbscan_columns,
    )


# ============================================================
# Run app
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)