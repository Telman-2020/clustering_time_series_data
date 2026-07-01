from pathlib import Path
import re

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

KMEANS_DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs.csv"
HDBSCAN_DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs_hdbscan.csv"


# ============================================================
# Features used for clustering
# ============================================================

FEATURES_USED_FOR_CLUSTERING = [
    "GX",
    "GY",
    "GZ",
    "BX",
    "BY",
    "BZ",
    "CC_RPM",
    "RPM_LT",
    "RPM_UT",
    "RX",
    "RY",
]


# ============================================================
# Data loading helpers
# ============================================================

def extract_job_number_from_source_file(source_file: str) -> int:
    """
    Extract job number from file names like:
    time_series_job1.csv
    """
    match = re.search(r"job(\d+)", str(source_file).lower())

    if match:
        return int(match.group(1))

    return -1


def load_clustered_data(path: Path, method_name: str) -> pd.DataFrame:
    """
    Load clustered result file and prepare helper columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {method_name} result file:\n{path}\n\n"
            "Run the clustering scripts first."
        )

    df = pd.read_csv(path)

    missing_features = [
        col for col in FEATURES_USED_FOR_CLUSTERING
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{method_name} file is missing required features: {missing_features}"
        )

    if "cluster" not in df.columns:
        raise ValueError(f"{method_name} file does not contain a 'cluster' column.")

    if "job_number" not in df.columns:
        if "source_file" not in df.columns:
            raise ValueError(
                f"{method_name} file needs either 'job_number' or 'source_file'."
            )

        df["job_number"] = df["source_file"].apply(extract_job_number_from_source_file)

    if "date_time_pd" in df.columns:
        df["date_time_pd"] = pd.to_datetime(df["date_time_pd"], errors="coerce")

    sort_columns = ["job_number"]

    if "date_time_pd" in df.columns and df["date_time_pd"].notna().sum() > 0:
        sort_columns.append("date_time_pd")
    elif "MD" in df.columns:
        sort_columns.append("MD")

    df = df.sort_values(sort_columns).copy()
    df["row_in_job"] = df.groupby("job_number").cumcount()

    df["method"] = method_name

    return df


print("Loading result CSV files...")

DATASETS = {
    "KMeans": load_clustered_data(KMEANS_DATA_PATH, "KMeans"),
    "HDBSCAN": load_clustered_data(HDBSCAN_DATA_PATH, "HDBSCAN"),
}

print("Loaded:")
for name, data in DATASETS.items():
    print(f" - {name}: {data.shape}")


# ============================================================
# App helpers
# ============================================================

def get_dataset(method: str) -> pd.DataFrame:
    return DATASETS[method].copy()


def get_available_x_axes(df: pd.DataFrame) -> list[dict]:
    options = [{"label": "Row number inside job", "value": "row_in_job"}]

    if "MD" in df.columns:
        options.append({"label": "Measured Depth / MD", "value": "MD"})

    if "date_time_pd" in df.columns and df["date_time_pd"].notna().sum() > 0:
        options.append({"label": "Date time", "value": "date_time_pd"})

    return options


def cluster_label(method: str, cluster_value: int) -> str:
    if method == "HDBSCAN" and cluster_value == -1:
        return "Noise / Outlier (-1)"

    return f"Cluster {cluster_value}"


def create_color_mapping(method: str, cluster_values: list[int]) -> dict[int, str]:
    """
    Create consistent cluster colors.
    HDBSCAN noise/outlier is black.
    """
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet

    normal_clusters = [c for c in cluster_values if c != -1]

    cluster_to_color = {}

    if method == "HDBSCAN" and -1 in cluster_values:
        cluster_to_color[-1] = "black"

    for i, cluster in enumerate(normal_clusters):
        cluster_to_color[cluster] = palette[i % len(palette)]

    return cluster_to_color


def downsample_by_position(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """
    Downsample rows by position to keep the app responsive.
    This preserves the shape across the full time/depth range.
    """
    if max_points is None or max_points <= 0:
        return df

    if len(df) <= max_points:
        return df

    positions = np.linspace(0, len(df) - 1, max_points).astype(int)

    return df.iloc[positions].copy()


def make_cluster_summary(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Build summary table for the filtered data.
    """
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("cluster")
        .agg(rows=("cluster", "size"))
        .reset_index()
    )

    summary["percent_of_filtered_data"] = 100 * summary["rows"] / len(df)
    summary["cluster_label"] = summary["cluster"].apply(
        lambda x: cluster_label(method, int(x))
    )

    if method == "KMeans":
        if "possible_outlier" in df.columns:
            outlier_summary = (
                df.groupby("cluster")["possible_outlier"]
                .sum()
                .reset_index()
                .rename(columns={"possible_outlier": "possible_outliers"})
            )
            summary = summary.merge(outlier_summary, on="cluster", how="left")

        if "cluster_distance" in df.columns:
            distance_summary = (
                df.groupby("cluster")["cluster_distance"]
                .mean()
                .reset_index()
                .rename(columns={"cluster_distance": "avg_cluster_distance"})
            )
            summary = summary.merge(distance_summary, on="cluster", how="left")

    if method == "HDBSCAN":
        summary["cluster_type"] = summary["cluster"].apply(
            lambda x: "noise_outlier" if int(x) == -1 else "cluster"
        )

        if "cluster_probability" in df.columns:
            probability_summary = (
                df.groupby("cluster")["cluster_probability"]
                .mean()
                .reset_index()
                .rename(columns={"cluster_probability": "avg_cluster_probability"})
            )
            summary = summary.merge(probability_summary, on="cluster", how="left")

            # Probability for noise is not a true cluster-membership probability.
            summary.loc[
                summary["cluster"] == -1,
                "avg_cluster_probability"
            ] = np.nan

    front_cols = ["cluster", "cluster_label", "rows", "percent_of_filtered_data"]
    other_cols = [col for col in summary.columns if col not in front_cols]

    summary = summary[front_cols + other_cols]

    return summary.sort_values("cluster").reset_index(drop=True)


def create_stacked_feature_figure(
    df: pd.DataFrame,
    method: str,
    selected_features: list[str],
    x_axis: str,
    selected_clusters: list[int],
    max_points: int,
) -> go.Figure:
    """
    Create vertically stacked interactive scatter plots.
    """
    if not selected_features:
        fig = go.Figure()
        fig.update_layout(title="Please select at least one feature.")
        return fig

    filtered = df[df["cluster"].isin(selected_clusters)].copy()

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for the selected filters.")
        return fig

    filtered = filtered.sort_values(["job_number", x_axis])
    filtered = downsample_by_position(filtered, max_points=max_points)

    cluster_values = sorted(filtered["cluster"].dropna().unique())
    cluster_values = [int(c) for c in cluster_values]

    cluster_to_color = create_color_mapping(method, cluster_values)

    fig = make_subplots(
        rows=len(selected_features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=selected_features,
    )

    for row_index, feature in enumerate(selected_features, start=1):
        for cluster_value in cluster_values:
            cluster_df = filtered[filtered["cluster"] == cluster_value]

            fig.add_trace(
                go.Scattergl(
                    x=cluster_df[x_axis],
                    y=cluster_df[feature],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cluster_to_color[cluster_value],
                        opacity=0.75,
                    ),
                    name=cluster_label(method, cluster_value),
                    legendgroup=str(cluster_value),
                    showlegend=(row_index == 1),
                    hovertemplate=(
                        "Feature: " + feature +
                        "<br>Value: %{y}" +
                        "<br>X: %{x}" +
                        f"<br>{cluster_label(method, cluster_value)}" +
                        "<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )

        fig.update_yaxes(title_text=feature, row=row_index, col=1)

    fig.update_xaxes(title_text=x_axis, row=len(selected_features), col=1)

    fig.update_layout(
        title=f"{method} clustered time-series view",
        height=max(350, 260 * len(selected_features)),
        template="plotly_white",
        legend_title="Cluster",
        margin=dict(l=70, r=30, t=80, b=50),
    )

    return fig


# ============================================================
# Dash app
# ============================================================

app = Dash(__name__)

initial_method = "HDBSCAN"
initial_df = get_dataset(initial_method)

app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "margin": "20px",
        "maxWidth": "1500px",
    },
    children=[
        html.H1("Interactive Clustering Viewer"),

        html.P(
            "Use this app to inspect KMeans and HDBSCAN cluster results from the saved CSV files. "
            "Select the job, features, x-axis and clusters to display."
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Clustering method"),
                        dcc.Dropdown(
                            id="method-dropdown",
                            options=[
                                {"label": "KMeans", "value": "KMeans"},
                                {"label": "HDBSCAN", "value": "HDBSCAN"},
                            ],
                            value=initial_method,
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label("Job number"),
                        dcc.Dropdown(
                            id="job-dropdown",
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label("X-axis"),
                        dcc.Dropdown(
                            id="x-axis-dropdown",
                            clearable=False,
                        ),
                    ]
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 2fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Features to plot"),
                        dcc.Dropdown(
                            id="feature-dropdown",
                            options=[
                                {"label": feature, "value": feature}
                                for feature in FEATURES_USED_FOR_CLUSTERING
                            ],
                            value=["GX", "GY", "GZ", "CC_RPM", "RPM_LT", "RPM_UT"],
                            multi=True,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label("Clusters to display"),
                        dcc.Dropdown(
                            id="cluster-dropdown",
                            multi=True,
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        html.Label("Max points to display"),
                        dcc.Input(
                            id="max-points-input",
                            type="number",
                            value=12000,
                            min=1000,
                            step=1000,
                            style={"width": "100%"},
                        ),
                    ]
                ),
            ],
        ),

        dcc.Graph(
            id="cluster-graph",
            style={"width": "100%"},
        ),

        html.H2("Cluster summary for current selection"),

        dash_table.DataTable(
            id="summary-table",
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "6px",
                "fontFamily": "Arial",
                "fontSize": "13px",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f2f2f2",
            },
        ),
    ],
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("job-dropdown", "options"),
    Output("job-dropdown", "value"),
    Output("x-axis-dropdown", "options"),
    Output("x-axis-dropdown", "value"),
    Input("method-dropdown", "value"),
)
def update_job_and_x_axis_options(method):
    df = get_dataset(method)

    job_numbers = sorted(df["job_number"].dropna().unique())
    job_numbers = [int(job) for job in job_numbers]

    job_options = [{"label": "All jobs", "value": "all"}] + [
        {"label": f"Job {job}", "value": job}
        for job in job_numbers
    ]

    x_axis_options = get_available_x_axes(df)

    # Prefer MD if available, otherwise row number.
    x_axis_values = [option["value"] for option in x_axis_options]

    if "MD" in x_axis_values:
        default_x_axis = "MD"
    else:
        default_x_axis = "row_in_job"

    return job_options, "all", x_axis_options, default_x_axis


@app.callback(
    Output("cluster-dropdown", "options"),
    Output("cluster-dropdown", "value"),
    Input("method-dropdown", "value"),
    Input("job-dropdown", "value"),
)
def update_cluster_options(method, selected_job):
    df = get_dataset(method)

    if selected_job != "all":
        df = df[df["job_number"] == int(selected_job)]

    cluster_values = sorted(df["cluster"].dropna().unique())
    cluster_values = [int(c) for c in cluster_values]

    options = [
        {
            "label": cluster_label(method, cluster_value),
            "value": cluster_value,
        }
        for cluster_value in cluster_values
    ]

    return options, cluster_values


@app.callback(
    Output("cluster-graph", "figure"),
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Input("method-dropdown", "value"),
    Input("job-dropdown", "value"),
    Input("x-axis-dropdown", "value"),
    Input("feature-dropdown", "value"),
    Input("cluster-dropdown", "value"),
    Input("max-points-input", "value"),
)
def update_graph_and_table(
    method,
    selected_job,
    x_axis,
    selected_features,
    selected_clusters,
    max_points,
):
    df = get_dataset(method)

    if selected_job != "all":
        df = df[df["job_number"] == int(selected_job)]

    if not selected_clusters:
        selected_clusters = sorted(df["cluster"].dropna().unique())
        selected_clusters = [int(c) for c in selected_clusters]

    selected_clusters = [int(c) for c in selected_clusters]

    fig = create_stacked_feature_figure(
        df=df,
        method=method,
        selected_features=selected_features,
        x_axis=x_axis,
        selected_clusters=selected_clusters,
        max_points=int(max_points or 12000),
    )

    filtered_for_summary = df[df["cluster"].isin(selected_clusters)].copy()
    summary = make_cluster_summary(filtered_for_summary, method)

    # Round numeric columns for display.
    for col in summary.columns:
        if pd.api.types.is_float_dtype(summary[col]):
            summary[col] = summary[col].round(4)

    columns = [
        {"name": col, "id": col}
        for col in summary.columns
    ]

    return fig, summary.to_dict("records"), columns


# ============================================================
# Run app
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
