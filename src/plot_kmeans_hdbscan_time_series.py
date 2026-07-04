from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
FIGURE_OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"

KMEANS_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "kmeans_clustered_trip_features.parquet"
)

HDBSCAN_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "hdbscan_clustered_trip_features.parquet"
)


DEFAULT_FEATURES = [
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


def load_clustered_data(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a clustered Parquet dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Clustered dataset was not found:\n"
            f"{file_path}\n\n"
            "Run this command first:\n"
            "python src\\cluster_time_series_methods.py"
        )

    data = pd.read_parquet(
        file_path,
        engine="pyarrow",
    )

    if data.empty:
        raise ValueError(
            f"The dataset is empty: {file_path}"
        )

    if "cluster" not in data.columns:
        raise ValueError(
            f"The cluster column is missing from: {file_path}"
        )

    if "time" in data.columns:
        data["time"] = pd.to_datetime(
            data["time"],
            errors="coerce",
        )

    return data


def filter_trip(
    data: pd.DataFrame,
    trip_id: int | None,
) -> pd.DataFrame:
    """
    Filter one trip or keep all trips.
    """
    if trip_id is None:
        return data.copy()

    filtered = data[
        data["trip_id"] == int(trip_id)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows were found for trip {trip_id}."
        )

    return filtered


def downsample_data(
    data: pd.DataFrame,
    maximum_points: int,
) -> pd.DataFrame:
    """
    Uniformly downsample rows for faster plotting.
    """
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


def cluster_display_name(
    method: str,
    cluster: int,
) -> str:
    """
    Return a readable cluster label.
    """
    if method == "hdbscan" and cluster == -1:
        return "Anomaly / Noise"

    return f"Cluster {cluster}"


def validate_features(
    data: pd.DataFrame,
    requested_features: list[str],
) -> list[str]:
    """
    Keep only numeric features present in the dataset.
    """
    valid_features = [
        feature
        for feature in requested_features
        if feature in data.columns
        and pd.api.types.is_numeric_dtype(
            data[feature]
        )
    ]

    missing_features = [
        feature
        for feature in requested_features
        if feature not in valid_features
    ]

    if missing_features:
        print(
            "Skipping unavailable features:",
            missing_features,
        )

    if not valid_features:
        raise ValueError(
            "None of the requested plotting features are available."
        )

    return valid_features


def plot_clustered_features(
    data: pd.DataFrame,
    *,
    method: str,
    features: list[str],
    x_axis: str,
    trip_id: int | None,
    maximum_points: int,
) -> Path:
    """
    Plot multiple time-series features coloured by cluster.
    """
    if x_axis not in data.columns:
        raise ValueError(
            f"X-axis column is unavailable: {x_axis}"
        )

    data = data.sort_values(
        ["trip_id", x_axis],
        kind="stable",
    )

    data = downsample_data(
        data,
        maximum_points,
    )

    clusters = sorted(
        int(value)
        for value in data[
            "cluster"
        ].dropna().unique()
    )

    figure, axes = plt.subplots(
        nrows=len(features),
        ncols=1,
        figsize=(
            16,
            max(4, 3.2 * len(features)),
        ),
        sharex=True,
    )

    if len(features) == 1:
        axes = [axes]

    for axis, feature in zip(
        axes,
        features,
    ):
        for cluster in clusters:
            cluster_data = data[
                data["cluster"] == cluster
            ]

            axis.scatter(
                cluster_data[x_axis],
                cluster_data[feature],
                s=8,
                alpha=0.7,
                label=cluster_display_name(
                    method,
                    cluster,
                ),
            )

        axis.set_ylabel(feature)
        axis.grid(
            visible=True,
            alpha=0.25,
        )

    axes[-1].set_xlabel(x_axis)

    trip_text = (
        f"Trip {trip_id}"
        if trip_id is not None
        else "All trips"
    )

    figure.suptitle(
        f"{method.upper()} clustered sensor features — "
        f"{trip_text}"
    )

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="center right",
        title="Cluster",
    )

    figure.tight_layout(
        rect=(0, 0, 0.88, 0.97)
    )

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trip_suffix = (
        f"trip_{trip_id}"
        if trip_id is not None
        else "all_trips"
    )

    output_path = (
        FIGURE_OUTPUT_DIR
        / f"{method}_{trip_suffix}_{x_axis}.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot KMeans and HDBSCAN clustered "
            "time-series features."
        )
    )

    parser.add_argument(
        "--trip",
        type=int,
        default=None,
        help=(
            "Trip number to plot. "
            "Omit to plot all trips."
        ),
    )

    parser.add_argument(
        "--x-axis",
        default="depth",
        choices=[
            "row_in_trip",
            "depth",
            "time",
            "ground_depth",
            "ground_time",
        ],
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
    )

    parser.add_argument(
        "--maximum-points",
        type=int,
        default=12000,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    datasets = {
        "kmeans": load_clustered_data(
            KMEANS_DATA_PATH
        ),
        "hdbscan": load_clustered_data(
            HDBSCAN_DATA_PATH
        ),
    }

    print("Creating clustering figures...")

    for method, data in datasets.items():
        data = filter_trip(
            data,
            arguments.trip,
        )

        features = validate_features(
            data,
            arguments.features,
        )

        output_path = plot_clustered_features(
            data,
            method=method,
            features=features,
            x_axis=arguments.x_axis,
            trip_id=arguments.trip,
            maximum_points=arguments.maximum_points,
        )

        print(
            f"{method.upper()} figure saved to:\n"
            f"{output_path}\n"
        )


if __name__ == "__main__":
    main()