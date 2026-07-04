from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from prepared_data import (
    DEFAULT_CLUSTERING_FEATURES,
    get_available_numeric_features,
    load_prepared_data,
    validate_default_features,
)


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

KMEANS_OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "kmeans_clustered_trip_features.parquet"
)

HDBSCAN_OUTPUT_PATH = (
    PROCESSED_DATA_DIR
    / "hdbscan_clustered_trip_features.parquet"
)

KMEANS_SUMMARY_PATH = (
    OUTPUTS_DIR
    / "kmeans_cluster_summary.csv"
)

HDBSCAN_SUMMARY_PATH = (
    OUTPUTS_DIR
    / "hdbscan_cluster_summary.csv"
)

COMPARISON_SUMMARY_PATH = (
    OUTPUTS_DIR
    / "clustering_method_comparison.csv"
)


# ============================================================
# Default model configuration
# ============================================================

DEFAULT_KMEANS_CLUSTERS = 7
DEFAULT_KMEANS_MAX_ITER = 100
DEFAULT_KMEANS_TOLERANCE = 1e-4
DEFAULT_KMEANS_RANDOM_STATE = 43
DEFAULT_KMEANS_OUTLIER_QUANTILE = 0.98

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 300
DEFAULT_HDBSCAN_MIN_SAMPLES = 50
DEFAULT_HDBSCAN_METRIC = "euclidean"
DEFAULT_HDBSCAN_SELECTION_METHOD = "eom"
DEFAULT_HDBSCAN_SELECTION_EPSILON = 0.0

# Six normal regimes plus noise label -1.
DEFAULT_HDBSCAN_TOTAL_LABELS = 7


# ============================================================
# Feature preparation
# ============================================================

def robust_clip(
    feature_data: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clip extreme values independently for each selected feature.

    Missing values are not filled.
    """
    lower_limits = feature_data.quantile(lower_quantile)
    upper_limits = feature_data.quantile(upper_quantile)

    return feature_data.clip(
        lower=lower_limits,
        upper=upper_limits,
        axis=1,
    )


def standardize(
    feature_data: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Standardise selected features.

    Returns:
    - scaled NumPy matrix
    - feature means
    - feature standard deviations
    """
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(feature_data)

    means = pd.Series(
        scaler.mean_,
        index=feature_data.columns,
        name="mean",
    )

    standard_deviations = pd.Series(
        scaler.scale_,
        index=feature_data.columns,
        name="standard_deviation",
    )

    return scaled_data, means, standard_deviations


def prepare_selected_feature_matrix(
    data: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """
    Prepare one shared model matrix for KMeans and HDBSCAN.

    No interpolation, forward filling, backward filling, or median
    imputation is performed.

    Only rows containing finite values for every selected feature
    are used by the clustering algorithms.
    """
    if not selected_features:
        raise ValueError(
            "At least one clustering feature must be selected."
        )

    available_features = get_available_numeric_features(data)

    unavailable_features = [
        feature
        for feature in selected_features
        if feature not in available_features
    ]

    if unavailable_features:
        raise ValueError(
            "Selected features are unavailable: "
            f"{unavailable_features}"
        )

    feature_data = data[selected_features].copy()

    for feature in selected_features:
        feature_data[feature] = pd.to_numeric(
            feature_data[feature],
            errors="coerce",
        )

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_row_mask = feature_data.notna().all(axis=1)

    eligible_data = (
        data.loc[valid_row_mask]
        .copy()
        .reset_index(drop=True)
    )

    eligible_features = (
        feature_data.loc[valid_row_mask]
        .copy()
        .reset_index(drop=True)
    )

    if eligible_features.empty:
        raise ValueError(
            "No rows contain valid measurements for all selected "
            "features."
        )

    clipped_features = robust_clip(eligible_features)

    scaled_features, _, _ = standardize(clipped_features)

    metadata = {
        "selected_features": selected_features,
        "raw_rows": int(len(data)),
        "eligible_rows": int(len(eligible_data)),
        "excluded_rows": int(len(data) - len(eligible_data)),
        "eligible_percent": (
            100.0 * len(eligible_data) / max(len(data), 1)
        ),
    }

    return eligible_data, scaled_features, metadata


# ============================================================
# KMeans
# ============================================================

def kmeans_numpy(
    feature_matrix: np.ndarray,
    k: int,
    max_iter: int = DEFAULT_KMEANS_MAX_ITER,
    tol: float = DEFAULT_KMEANS_TOLERANCE,
    random_state: int = DEFAULT_KMEANS_RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray,
]:
    """
    Run KMeans and return a structure compatible with the dashboard.

    Returns:
    - labels
    - centroids
    - inertia
    - number of iterations
    - squared distance from each row to its assigned centroid
    """
    if k < 2:
        raise ValueError("KMeans k must be at least 2.")

    if k >= len(feature_matrix):
        raise ValueError(
            "KMeans k must be smaller than the eligible row count."
        )

    model = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=10,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
    )

    labels = model.fit_predict(feature_matrix)

    centroids = model.cluster_centers_

    assigned_centroids = centroids[labels]

    squared_distances = np.sum(
        (feature_matrix - assigned_centroids) ** 2,
        axis=1,
    )

    return (
        labels,
        centroids,
        float(model.inertia_),
        int(model.n_iter_),
        squared_distances,
    )


def run_kmeans(
    data: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    n_clusters: int = DEFAULT_KMEANS_CLUSTERS,
    max_iter: int = DEFAULT_KMEANS_MAX_ITER,
    tolerance: float = DEFAULT_KMEANS_TOLERANCE,
    random_state: int = DEFAULT_KMEANS_RANDOM_STATE,
    outlier_quantile: float = DEFAULT_KMEANS_OUTLIER_QUANTILE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run KMeans and add clustering results to the eligible rows.
    """
    (
        labels,
        _,
        inertia,
        iterations,
        squared_distances,
    ) = kmeans_numpy(
        feature_matrix,
        k=n_clusters,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
    )

    result = data.copy()

    result["cluster"] = labels

    result["cluster_distance"] = np.sqrt(
        squared_distances
    )

    outlier_threshold = (
        result["cluster_distance"]
        .quantile(outlier_quantile)
    )

    result["possible_outlier"] = (
        result["cluster_distance"]
        >= outlier_threshold
    )

    metadata = {
        "method": "KMeans",
        "clusters": int(n_clusters),
        "rows": int(len(result)),
        "inertia": float(inertia),
        "iterations": int(iterations),
        "outlier_threshold": float(outlier_threshold),
        "outlier_rows": int(
            result["possible_outlier"].sum()
        ),
    }

    return result, metadata


# ============================================================
# HDBSCAN
# ============================================================

def run_hdbscan_model(
    feature_matrix: np.ndarray,
    *,
    min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    metric: str = DEFAULT_HDBSCAN_METRIC,
    cluster_selection_method: str = (
        DEFAULT_HDBSCAN_SELECTION_METHOD
    ),
    cluster_selection_epsilon: float = (
        DEFAULT_HDBSCAN_SELECTION_EPSILON
    ),
) -> HDBSCAN:
    """
    Fit HDBSCAN using the selected hyperparameters.
    """
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=str(metric),
        cluster_selection_method=str(
            cluster_selection_method
        ),
        cluster_selection_epsilon=float(
            cluster_selection_epsilon
        ),
        n_jobs=-1,
    )

    model.fit(feature_matrix)

    return model


def merge_hdbscan_clusters_to_target(
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    target_normal_clusters: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Merge HDBSCAN normal clusters when more than the target exist.

    Noise label -1 is never changed.
    """
    labels = np.asarray(labels, dtype=int)

    normal_labels = sorted(
        int(label)
        for label in np.unique(labels)
        if int(label) != -1
    )

    if len(normal_labels) <= target_normal_clusters:
        return labels.copy(), {
            label: label
            for label in normal_labels
        }

    centroids = []

    for label in normal_labels:
        centroids.append(
            feature_matrix[labels == label].mean(axis=0)
        )

    centroid_matrix = np.vstack(centroids)

    merger = AgglomerativeClustering(
        n_clusters=int(target_normal_clusters),
        linkage="ward",
    )

    merged_centroid_labels = merger.fit_predict(
        centroid_matrix
    )

    merge_map = {
        original_label: int(merged_label)
        for original_label, merged_label in zip(
            normal_labels,
            merged_centroid_labels,
        )
    }

    merged_labels = np.full(
        labels.shape,
        -1,
        dtype=int,
    )

    for original_label, merged_label in merge_map.items():
        merged_labels[
            labels == original_label
        ] = merged_label

    return merged_labels, merge_map


def split_hdbscan_clusters_to_target(
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    target_normal_clusters: int,
) -> np.ndarray:
    """
    Split the largest normal clusters if HDBSCAN returns fewer than
    the requested number of normal regimes.

    Noise label -1 remains untouched.
    """
    result = np.asarray(labels, dtype=int).copy()

    while True:
        normal_labels = sorted(
            int(label)
            for label in np.unique(result)
            if int(label) != -1
        )

        if len(normal_labels) >= target_normal_clusters:
            break

        cluster_sizes = {
            label: int(np.sum(result == label))
            for label in normal_labels
        }

        candidate_labels = sorted(
            normal_labels,
            key=lambda label: cluster_sizes[label],
            reverse=True,
        )

        split_completed = False

        for label in candidate_labels:
            row_indices = np.flatnonzero(
                result == label
            )

            if len(row_indices) < 2:
                continue

            cluster_points = feature_matrix[
                row_indices
            ]

            split_model = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=(
                    1000 + len(normal_labels)
                ),
            )

            split_labels = split_model.fit_predict(
                cluster_points
            )

            if len(np.unique(split_labels)) < 2:
                continue

            new_label = max(normal_labels) + 1

            result[
                row_indices[split_labels == 1]
            ] = new_label

            split_completed = True
            break

        if not split_completed:
            break

    normal_labels = sorted(
        int(label)
        for label in np.unique(result)
        if int(label) != -1
    )

    relabel_map = {
        old_label: new_label
        for new_label, old_label in enumerate(
            normal_labels
        )
    }

    cleaned_labels = np.full(
        result.shape,
        -1,
        dtype=int,
    )

    for old_label, new_label in relabel_map.items():
        cleaned_labels[
            result == old_label
        ] = new_label

    return cleaned_labels


def make_hdbscan_target_labels(
    original_labels: np.ndarray,
    feature_matrix: np.ndarray,
    target_total_labels: int,
) -> np.ndarray:
    """
    Produce the requested displayed label count.

    One label is reserved for Anomaly / Noise (-1).
    """
    target_normal_clusters = max(
        int(target_total_labels) - 1,
        1,
    )

    merged_labels, _ = (
        merge_hdbscan_clusters_to_target(
            labels=original_labels,
            feature_matrix=feature_matrix,
            target_normal_clusters=(
                target_normal_clusters
            ),
        )
    )

    current_normal_count = len(
        [
            label
            for label in np.unique(merged_labels)
            if int(label) != -1
        ]
    )

    if current_normal_count < target_normal_clusters:
        merged_labels = (
            split_hdbscan_clusters_to_target(
                labels=merged_labels,
                feature_matrix=feature_matrix,
                target_normal_clusters=(
                    target_normal_clusters
                ),
            )
        )

    return merged_labels


def run_hdbscan(
    data: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    min_cluster_size: int = (
        DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
    ),
    min_samples: int = (
        DEFAULT_HDBSCAN_MIN_SAMPLES
    ),
    metric: str = DEFAULT_HDBSCAN_METRIC,
    cluster_selection_method: str = (
        DEFAULT_HDBSCAN_SELECTION_METHOD
    ),
    cluster_selection_epsilon: float = (
        DEFAULT_HDBSCAN_SELECTION_EPSILON
    ),
    target_total_labels: int = (
        DEFAULT_HDBSCAN_TOTAL_LABELS
    ),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run HDBSCAN and preserve its native noise label -1.
    """
    model = run_hdbscan_model(
        feature_matrix,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=(
            cluster_selection_method
        ),
        cluster_selection_epsilon=(
            cluster_selection_epsilon
        ),
    )

    original_labels = np.asarray(
        model.labels_,
        dtype=int,
    )

    displayed_labels = make_hdbscan_target_labels(
        original_labels=original_labels,
        feature_matrix=feature_matrix,
        target_total_labels=target_total_labels,
    )

    result = data.copy()

    result["hdbscan_original_cluster"] = (
        original_labels
    )

    result["cluster"] = displayed_labels

    result["hdbscan_noise_outlier"] = (
        original_labels == -1
    )

    if hasattr(model, "probabilities_"):
        result["cluster_probability"] = (
            model.probabilities_
        )
    else:
        result["cluster_probability"] = np.nan

    noise_rows = int(
        (original_labels == -1).sum()
    )

    metadata = {
        "method": "HDBSCAN",
        "rows": int(len(result)),
        "native_normal_clusters": int(
            pd.Series(
                original_labels[
                    original_labels != -1
                ]
            ).nunique()
        ),
        "displayed_normal_clusters": int(
            result.loc[
                result["cluster"] != -1,
                "cluster",
            ].nunique()
        ),
        "displayed_total_labels": int(
            result["cluster"].nunique()
        ),
        "noise_rows": noise_rows,
        "noise_percent": (
            100.0 * noise_rows / max(len(result), 1)
        ),
    }

    return result, metadata


# ============================================================
# Summaries
# ============================================================

def create_kmeans_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
            possible_outliers=(
                "possible_outlier",
                "sum",
            ),
            average_cluster_distance=(
                "cluster_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["percent_of_data"] = (
        100.0 * summary["rows"] / len(data)
    )

    return summary


def create_hdbscan_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
            average_membership_probability=(
                "cluster_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["cluster_label"] = summary[
        "cluster"
    ].apply(
        lambda value: (
            "Anomaly / Noise"
            if int(value) == -1
            else f"Cluster {int(value)}"
        )
    )

    summary["percent_of_data"] = (
        100.0 * summary["rows"] / len(data)
    )

    summary.loc[
        summary["cluster"] == -1,
        "average_membership_probability",
    ] = np.nan

    return summary


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared Parquet dataset...")

    prepared_data = load_prepared_data()

    validate_default_features(prepared_data)

    print("Prepared data shape:", prepared_data.shape)
    print("Trips:", prepared_data["trip_id"].nunique())

    print("\nDefault clustering features:")

    for feature in DEFAULT_CLUSTERING_FEATURES:
        print(f" - {feature}")

    (
        eligible_data,
        feature_matrix,
        preparation_metadata,
    ) = prepare_selected_feature_matrix(
        data=prepared_data,
        selected_features=DEFAULT_CLUSTERING_FEATURES,
    )

    print("\nFeature preparation summary:")
    print(
        f"Raw rows: "
        f"{preparation_metadata['raw_rows']:,}"
    )
    print(
        f"Eligible rows: "
        f"{preparation_metadata['eligible_rows']:,}"
    )
    print(
        f"Excluded rows: "
        f"{preparation_metadata['excluded_rows']:,}"
    )

    print("\nRunning KMeans...")

    kmeans_data, kmeans_metadata = run_kmeans(
        data=eligible_data,
        feature_matrix=feature_matrix,
    )

    print("Running HDBSCAN...")

    hdbscan_data, hdbscan_metadata = run_hdbscan(
        data=eligible_data,
        feature_matrix=feature_matrix,
    )

    kmeans_summary = create_kmeans_summary(
        kmeans_data
    )

    hdbscan_summary = create_hdbscan_summary(
        hdbscan_data
    )

    comparison_summary = pd.DataFrame(
        [
            {
                **preparation_metadata,
                **kmeans_metadata,
            },
            {
                **preparation_metadata,
                **hdbscan_metadata,
            },
        ]
    )

    kmeans_data.to_parquet(
        KMEANS_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    hdbscan_data.to_parquet(
        HDBSCAN_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    kmeans_summary.to_csv(
        KMEANS_SUMMARY_PATH,
        index=False,
    )

    hdbscan_summary.to_csv(
        HDBSCAN_SUMMARY_PATH,
        index=False,
    )

    comparison_summary.to_csv(
        COMPARISON_SUMMARY_PATH,
        index=False,
    )

    print("\nPipeline completed.")

    print("\nKMeans result:")
    print(KMEANS_OUTPUT_PATH)

    print("\nHDBSCAN result:")
    print(HDBSCAN_OUTPUT_PATH)

    print("\nKMeans summary:")
    print(kmeans_summary)

    print("\nHDBSCAN summary:")
    print(hdbscan_summary)

    print(
        "\nHDBSCAN Anomaly / Noise rows:",
        hdbscan_metadata["noise_rows"],
    )

    print(
        "HDBSCAN Anomaly / Noise percentage:",
        round(
            hdbscan_metadata["noise_percent"],
            2,
        ),
        "%",
    )


if __name__ == "__main__":
    main()