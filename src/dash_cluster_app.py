from __future__ import annotations

import re
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots
from sklearn.cluster import HDBSCAN

from cluster_time_series_methods import (
    RAW_DATA_DIR,
    kmeans_numpy,
    load_all_csv_files,
    merge_hdbscan_clusters_to_target,
    robust_clip,
    standardize,
)


# ============================================================
# Configuration
# ============================================================

KMEANS_METHOD = "KMeans"
HDBSCAN_METHOD = "HDBSCAN"

# Default values shown in the tuning panel.
DEFAULT_KMEANS_N_CLUSTERS = 7
DEFAULT_KMEANS_MAX_ITER = 100
DEFAULT_KMEANS_TOLERANCE = 1e-4
DEFAULT_KMEANS_RANDOM_STATE = 43
DEFAULT_KMEANS_OUTLIER_QUANTILE = 0.98

DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 300
DEFAULT_HDBSCAN_MIN_SAMPLES = 50
DEFAULT_HDBSCAN_METRIC = "euclidean"
DEFAULT_HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
DEFAULT_HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# Includes the preserved Anomaly / Noise label -1.
DEFAULT_HDBSCAN_TARGET_TOTAL_LABELS = 7

# Raw columns requested for the proof-of-concept. Column names are normalised
# to lower case before this mapping is applied.
RAW_TO_CANONICAL_COLUMN_NAMES = {
    "md": "depth",
    "date_time_pd": "time",
    "ct_inc": "cont_vertical_orientation_incl",
    "ct_azi": "cont_horizontal_orientation_azim",
    "st_inc": "static_vertical_orientation_incl",
    "st_azi": "static_horizontal_orientation_azim",
    "cc_rpm": "pipe_rotation",
    "rpm_lt": "flow_rotation_lower",
    "rpm_ut": "flow_rotation_upper",
    "mwd_md": "ground_depth",
    "date_time_mwd": "ground_time",
    "mwd_incl": "ground_vertical_orientation",
    "mwd_azim": "ground_horizontal_orientation",
}

# All requested raw and engineered features made available in the dashboard.
REQUESTED_FEATURES = [
    "depth",
    "time",
    "cont_vertical_orientation_incl",
    "cont_horizontal_orientation_azim",
    "static_vertical_orientation_incl",
    "static_horizontal_orientation_azim",
    "gx",
    "gy",
    "gz",
    "bx",
    "by",
    "bz",
    "pipe_rotation",
    "flow_rotation_lower",
    "flow_rotation_upper",
    "rx",
    "ry",
    "axlshkpeak",
    "axlshkrms",
    "radshkrms",
    "radshkpeak",
    "ground_depth",
    "ground_time",
    "ground_vertical_orientation",
    "ground_horizontal_orientation",
    "azim_raw",
    "azim_abs_difference",
]

# Default clustering inputs requested by the user: accelerometer,
# magnetometer, electronic rotation, pipe rotation and flow rotation.
DEFAULT_CLUSTERING_FEATURES = [
    "gx",
    "gy",
    "gz",
    "bx",
    "by",
    "bz",
    "rx",
    "ry",
    "pipe_rotation",
    "flow_rotation_lower",
    "flow_rotation_upper",
]

DEFAULT_PLOT_FEATURES = [
    "gx",
    "gy",
    "gz",
    "bx",
    "by",
    "bz",
    "pipe_rotation",
    "flow_rotation_lower",
    "flow_rotation_upper",
    "azim_raw",
    "azim_abs_difference",
]

TIME_ORIGIN = pd.Timestamp("2020-01-01")


# ============================================================
# Raw-data preparation
# ============================================================


def extract_job_number_from_source_file(source_file: str) -> int:
    """Extract a job number from names such as time_series_job12.csv."""
    match = re.search(r"job(\d+)", str(source_file).lower())
    return int(match.group(1)) if match else -1


def normalise_column_name(column_name: object) -> str:
    """Convert a raw column name to lower snake case."""
    normalised = str(column_name).strip().lower()
    normalised = re.sub(r"[^a-z0-9]+", "_", normalised)
    return normalised.strip("_")


def shift_datetime_columns_to_origin(
    data: pd.DataFrame,
    datetime_columns: list[str],
    origin: pd.Timestamp = TIME_ORIGIN,
) -> pd.DataFrame:
    """
    Shift all available datetime streams by one shared offset so the earliest
    valid timestamp starts at 2020-01-01.

    A shared offset preserves the relative timing between the higher-rate
    continuous stream and lower-rate MWD/S&V stream. No interpolation,
    forward-fill or backward-fill is applied.
    """
    result = data.copy()
    available_columns = [
        column for column in datetime_columns if column in result.columns
    ]

    valid_minima: list[pd.Timestamp] = []

    for column in available_columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        column_minimum = result[column].min()
        if pd.notna(column_minimum):
            valid_minima.append(pd.Timestamp(column_minimum))

    if not valid_minima:
        return result

    common_start = min(valid_minima)
    offset = origin - common_start

    for column in available_columns:
        result[column] = result[column] + offset

    return result


def compute_azim_raw(data: pd.DataFrame) -> pd.Series:
    """
    Compute raw azimuth from accelerometer and magnetometer vectors.

    The calculation is evaluated only where GX, GY, GZ, BX, BY and BZ are
    present and finite. Missing lower-rate observations remain missing.
    """
    required = ["gx", "gy", "gz", "bx", "by", "bz"]

    if any(column not in data.columns for column in required):
        return pd.Series(np.nan, index=data.index, dtype=float)

    vectors = data[required].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(vectors.to_numpy(dtype=float)).all(axis=1)

    azimuth = pd.Series(np.nan, index=data.index, dtype=float)

    if not valid_mask.any():
        return azimuth

    valid = vectors.loc[valid_mask]

    gx = valid["gx"].to_numpy(dtype=float)
    gy = valid["gy"].to_numpy(dtype=float)
    gz = valid["gz"].to_numpy(dtype=float)
    bx = valid["bx"].to_numpy(dtype=float)
    by = valid["by"].to_numpy(dtype=float)
    bz = valid["bz"].to_numpy(dtype=float)

    gravity_total = np.sqrt(gx**2 + gy**2 + gz**2)

    numerator = (gx * by - gy * bx) * gravity_total
    denominator = (
        bz * (gx**2 + gy**2)
        - gz * (gx * bx + gy * by)
    )

    raw_values = np.mod(
        np.degrees(np.arctan2(numerator, denominator)),
        360.0,
    )

    azimuth.loc[valid_mask] = raw_values
    return azimuth


def compute_absolute_azimuth_difference(data: pd.DataFrame) -> pd.Series:
    """
    Compute the absolute circular difference between MWD azimuth and azim_raw.

    Circular distance is used so 359° and 1° differ by 2°, not 358°.
    """
    required = ["ground_horizontal_orientation", "azim_raw"]

    if any(column not in data.columns for column in required):
        return pd.Series(np.nan, index=data.index, dtype=float)

    ground_azimuth = pd.to_numeric(
        data["ground_horizontal_orientation"],
        errors="coerce",
    )
    raw_azimuth = pd.to_numeric(data["azim_raw"], errors="coerce")

    circular_difference = (
        (ground_azimuth - raw_azimuth + 180.0) % 360.0
    ) - 180.0

    return circular_difference.abs()


def prepare_raw_data() -> pd.DataFrame:
    """
    Load, normalise and enrich the raw files used by the dashboard.

    Important:
    - all column names are lower case;
    - requested columns are renamed to domain-readable names;
    - both datetime streams are shifted with one common offset;
    - no interpolation or imputation is performed;
    - azim_raw and azim_abs_difference are engineered only where source
      measurements are genuinely available.
    """
    data = load_all_csv_files(RAW_DATA_DIR)

    data.columns = [
        normalise_column_name(column)
        for column in data.columns
    ]

    if data.columns.duplicated().any():
        duplicate_names = sorted(
            set(data.columns[data.columns.duplicated()].tolist())
        )
        raise ValueError(
            "Duplicate columns after lower-case normalisation: "
            f"{duplicate_names}"
        )

    data = data.rename(columns=RAW_TO_CANONICAL_COLUMN_NAMES)

    if "job_number" not in data.columns:
        if "source_file" not in data.columns:
            raise ValueError("Raw data requires 'job_number' or 'source_file'.")
        data["job_number"] = data["source_file"].apply(
            extract_job_number_from_source_file
        )

    data = shift_datetime_columns_to_origin(
        data,
        datetime_columns=["time", "ground_time"],
    )

    data["azim_raw"] = compute_azim_raw(data)
    data["azim_abs_difference"] = compute_absolute_azimuth_difference(data)

    sort_columns = ["job_number"]

    if "time" in data.columns and data["time"].notna().any():
        sort_columns.append("time")
    elif "depth" in data.columns:
        sort_columns.append("depth")

    data = data.sort_values(sort_columns).reset_index(drop=True)
    data["row_in_job"] = data.groupby("job_number").cumcount()

    return data


print("Loading raw data for the comparison dashboard...")
RAW_DATA = prepare_raw_data()

AVAILABLE_FEATURES = [
    feature
    for feature in REQUESTED_FEATURES
    if feature in RAW_DATA.columns
]

if not AVAILABLE_FEATURES:
    raise ValueError(
        "None of the requested raw or engineered features exists in the data."
    )

DEFAULT_MODEL_FEATURES = [
    feature
    for feature in DEFAULT_CLUSTERING_FEATURES
    if feature in AVAILABLE_FEATURES
]

if not DEFAULT_MODEL_FEATURES:
    raise ValueError(
        "None of the requested default clustering features exists in the data."
    )

DEFAULT_VISIBLE_FEATURES = [
    feature
    for feature in DEFAULT_PLOT_FEATURES
    if feature in AVAILABLE_FEATURES
]

if not DEFAULT_VISIBLE_FEATURES:
    DEFAULT_VISIBLE_FEATURES = AVAILABLE_FEATURES[: min(6, len(AVAILABLE_FEATURES))]


# Module-level cache suitable for this local proof-of-concept dashboard.
DATASETS: dict[str, pd.DataFrame] = {}
LAST_RUN_METADATA: dict[str, Any] = {}


# ============================================================
# Dynamic clustering
# ============================================================


def prepare_selected_feature_matrix(
    selected_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """
    Prepare one shared complete-case feature matrix for both algorithms.

    No interpolation, forward-fill, backward-fill or median imputation is
    performed. When a lower-rate feature is selected, only rows containing
    real finite observations for every selected feature are modelled.

    Datetime features are converted to elapsed seconds from 2020-01-01 for
    modelling while the original datetime columns remain available for plots.
    """
    if not selected_features:
        raise ValueError("Select at least one input feature.")

    invalid_features = [
        feature
        for feature in selected_features
        if feature not in AVAILABLE_FEATURES
    ]

    if invalid_features:
        raise ValueError(f"Unavailable features selected: {invalid_features}")

    X_df = pd.DataFrame(index=RAW_DATA.index)

    for feature in selected_features:
        if pd.api.types.is_datetime64_any_dtype(RAW_DATA[feature]):
            elapsed_seconds = (
                RAW_DATA[feature] - TIME_ORIGIN
            ).dt.total_seconds()
            X_df[feature] = elapsed_seconds
        else:
            X_df[feature] = pd.to_numeric(
                RAW_DATA[feature],
                errors="coerce",
            )

    numeric_values = X_df.to_numpy(dtype=float)
    valid_mask = np.isfinite(numeric_values).all(axis=1)

    model_data = RAW_DATA.loc[valid_mask].copy()
    model_features = X_df.loc[valid_mask].copy()

    if model_data.empty:
        raise ValueError(
            "No rows contain real finite values for every selected feature. "
            "Choose fewer lower-rate features or review the source data."
        )

    X_clipped = robust_clip(
        model_features,
        lower_q=0.01,
        upper_q=0.99,
    )
    X_scaled, _, _ = standardize(X_clipped)

    preparation_metadata = {
        "raw_rows": int(len(RAW_DATA)),
        "modelled_rows": int(len(model_data)),
        "excluded_rows": int(len(RAW_DATA) - len(model_data)),
    }

    return model_data, X_scaled, preparation_metadata


def split_normal_clusters_to_target(
    labels: np.ndarray,
    X_scaled: np.ndarray,
    target_normal_clusters: int,
) -> np.ndarray:
    """
    If HDBSCAN returns fewer than the required normal regimes, split the
    largest normal clusters until the target is reached.

    The HDBSCAN noise label -1 is never changed. This is post-processing for
    a like-for-like seven-label comparison, not native HDBSCAN behaviour.
    """
    result = np.asarray(labels, dtype=int).copy()

    while True:
        normal_labels = sorted(
            int(label) for label in np.unique(result) if int(label) != -1
        )

        if len(normal_labels) >= target_normal_clusters:
            break

        candidate_labels = sorted(
            normal_labels,
            key=lambda label: int(np.sum(result == label)),
            reverse=True,
        )

        split_completed = False

        for cluster_label_value in candidate_labels:
            row_indices = np.flatnonzero(result == cluster_label_value)

            if len(row_indices) < 2:
                continue

            cluster_points = X_scaled[row_indices]

            split_labels, _, _, _, _ = kmeans_numpy(
                cluster_points,
                k=2,
                max_iter=100,
                tol=1e-4,
                random_state=1000 + len(normal_labels),
            )

            unique_split_labels = np.unique(split_labels)

            if len(unique_split_labels) < 2:
                # Deterministic fallback for duplicate/constant points.
                feature_variances = np.var(cluster_points, axis=0)
                split_feature = int(np.argmax(feature_variances))
                ordered_positions = np.argsort(cluster_points[:, split_feature])
                midpoint = len(ordered_positions) // 2

                if midpoint == 0 or midpoint == len(ordered_positions):
                    continue

                split_labels = np.zeros(len(row_indices), dtype=int)
                split_labels[ordered_positions[midpoint:]] = 1

            new_cluster_label = max(normal_labels) + 1
            result[row_indices[split_labels == 1]] = new_cluster_label
            split_completed = True
            break

        if not split_completed:
            break

    # Relabel normal clusters to a clean 0..n-1 range. Keep -1 untouched.
    normal_labels = sorted(
        int(label) for label in np.unique(result) if int(label) != -1
    )
    relabel_map = {
        old_label: new_label
        for new_label, old_label in enumerate(normal_labels)
    }

    relabelled = np.full(result.shape, -1, dtype=int)
    for old_label, new_label in relabel_map.items():
        relabelled[result == old_label] = new_label

    return relabelled


def make_hdbscan_target_labels(
    original_labels: np.ndarray,
    X_scaled: np.ndarray,
    target_normal_clusters: int,
) -> np.ndarray:
    """
    Produce the requested number of normal regimes while keeping HDBSCAN
    noise label -1 untouched.

    This target is post-processing for comparison. It is not a native HDBSCAN
    hyperparameter.
    """
    merged_labels, _ = merge_hdbscan_clusters_to_target(
        labels=original_labels,
        X_scaled=X_scaled,
        target_normal_clusters=target_normal_clusters,
    )

    current_normal_count = len(
        [label for label in np.unique(merged_labels) if int(label) != -1]
    )

    if current_normal_count < target_normal_clusters:
        merged_labels = split_normal_clusters_to_target(
            labels=merged_labels,
            X_scaled=X_scaled,
            target_normal_clusters=target_normal_clusters,
        )

    return merged_labels


def validate_hyperparameters(
    kmeans_n_clusters: int,
    kmeans_max_iter: int,
    kmeans_tolerance: float,
    kmeans_outlier_quantile: float,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_cluster_selection_epsilon: float,
    hdbscan_target_total_labels: int,
) -> None:
    """Validate user-entered hyperparameters before fitting models."""
    if kmeans_n_clusters < 2:
        raise ValueError("KMeans k must be at least 2.")
    if kmeans_n_clusters >= len(RAW_DATA):
        raise ValueError("KMeans k must be smaller than the number of rows.")
    if kmeans_max_iter < 1:
        raise ValueError("KMeans max_iter must be at least 1.")
    if kmeans_tolerance <= 0:
        raise ValueError("KMeans tolerance must be greater than 0.")
    if not 0.5 < kmeans_outlier_quantile < 1.0:
        raise ValueError("KMeans outlier quantile must be between 0.5 and 1.0.")
    if hdbscan_min_cluster_size < 2:
        raise ValueError("HDBSCAN min_cluster_size must be at least 2.")
    if hdbscan_min_samples < 1:
        raise ValueError("HDBSCAN min_samples must be at least 1.")
    if hdbscan_cluster_selection_epsilon < 0:
        raise ValueError("HDBSCAN cluster_selection_epsilon cannot be negative.")
    if hdbscan_target_total_labels < 2:
        raise ValueError("Target total labels must be at least 2.")


def run_algorithms_for_features(
    selected_features: list[str],
    *,
    kmeans_n_clusters: int = DEFAULT_KMEANS_N_CLUSTERS,
    kmeans_max_iter: int = DEFAULT_KMEANS_MAX_ITER,
    kmeans_tolerance: float = DEFAULT_KMEANS_TOLERANCE,
    kmeans_random_state: int = DEFAULT_KMEANS_RANDOM_STATE,
    kmeans_outlier_quantile: float = DEFAULT_KMEANS_OUTLIER_QUANTILE,
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    hdbscan_metric: str = DEFAULT_HDBSCAN_METRIC,
    hdbscan_cluster_selection_method: str = DEFAULT_HDBSCAN_CLUSTER_SELECTION_METHOD,
    hdbscan_cluster_selection_epsilon: float = DEFAULT_HDBSCAN_CLUSTER_SELECTION_EPSILON,
    hdbscan_target_total_labels: int = DEFAULT_HDBSCAN_TARGET_TOTAL_LABELS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Refit KMeans and HDBSCAN using selected features and hyperparameters."""
    validate_hyperparameters(
        kmeans_n_clusters=kmeans_n_clusters,
        kmeans_max_iter=kmeans_max_iter,
        kmeans_tolerance=kmeans_tolerance,
        kmeans_outlier_quantile=kmeans_outlier_quantile,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        hdbscan_target_total_labels=hdbscan_target_total_labels,
    )

    started_at = time.perf_counter()
    data, X_scaled, preparation_metadata = prepare_selected_feature_matrix(
        selected_features
    )

    if int(kmeans_n_clusters) >= len(data):
        raise ValueError(
            "KMeans k must be smaller than the number of complete-case rows "
            f"({len(data):,})."
        )

    if int(hdbscan_min_cluster_size) > len(data):
        raise ValueError(
            "HDBSCAN min_cluster_size cannot exceed the number of "
            f"complete-case rows ({len(data):,})."
        )

    if int(hdbscan_min_samples) > len(data):
        raise ValueError(
            "HDBSCAN min_samples cannot exceed the number of complete-case "
            f"rows ({len(data):,})."
        )

    # ----------------------- KMeans ---------------------------
    (
        kmeans_labels,
        _,
        kmeans_inertia,
        kmeans_iterations,
        kmeans_distances_squared,
    ) = kmeans_numpy(
        X_scaled,
        k=int(kmeans_n_clusters),
        max_iter=int(kmeans_max_iter),
        tol=float(kmeans_tolerance),
        random_state=int(kmeans_random_state),
    )

    kmeans_result = data.copy()
    kmeans_result["method"] = KMEANS_METHOD
    kmeans_result["cluster"] = kmeans_labels
    kmeans_result["cluster_distance"] = np.sqrt(kmeans_distances_squared)

    kmeans_outlier_threshold = kmeans_result["cluster_distance"].quantile(
        float(kmeans_outlier_quantile)
    )
    kmeans_result["possible_outlier"] = (
        kmeans_result["cluster_distance"] >= kmeans_outlier_threshold
    )

    # ----------------------- HDBSCAN --------------------------
    hdbscan_model = HDBSCAN(
        min_cluster_size=int(hdbscan_min_cluster_size),
        min_samples=int(hdbscan_min_samples),
        metric=str(hdbscan_metric),
        cluster_selection_method=str(hdbscan_cluster_selection_method),
        cluster_selection_epsilon=float(hdbscan_cluster_selection_epsilon),
        n_jobs=-1,
    )
    hdbscan_model.fit(X_scaled)
    original_hdbscan_labels = np.asarray(hdbscan_model.labels_, dtype=int)

    # Reserve one label for the untouched HDBSCAN noise class -1.
    target_normal_clusters = max(int(hdbscan_target_total_labels) - 1, 1)
    final_hdbscan_labels = make_hdbscan_target_labels(
        original_labels=original_hdbscan_labels,
        X_scaled=X_scaled,
        target_normal_clusters=target_normal_clusters,
    )

    hdbscan_result = data.copy()
    hdbscan_result["method"] = HDBSCAN_METHOD
    hdbscan_result["hdbscan_original_cluster"] = original_hdbscan_labels
    hdbscan_result["cluster"] = final_hdbscan_labels
    hdbscan_result["hdbscan_noise_outlier"] = original_hdbscan_labels == -1

    if hasattr(hdbscan_model, "probabilities_"):
        hdbscan_result["cluster_probability"] = hdbscan_model.probabilities_
    else:
        hdbscan_result["cluster_probability"] = np.nan

    elapsed_seconds = time.perf_counter() - started_at
    noise_rows = int((hdbscan_result["cluster"] == -1).sum())
    original_normal_clusters = int(
        pd.Series(original_hdbscan_labels[original_hdbscan_labels != -1]).nunique()
    )

    metadata = {
        "features": selected_features,
        "raw_rows": preparation_metadata["raw_rows"],
        "modelled_rows": preparation_metadata["modelled_rows"],
        "excluded_rows": preparation_metadata["excluded_rows"],
        "elapsed_seconds": elapsed_seconds,
        "kmeans_n_clusters": int(kmeans_n_clusters),
        "kmeans_max_iter": int(kmeans_max_iter),
        "kmeans_tolerance": float(kmeans_tolerance),
        "kmeans_random_state": int(kmeans_random_state),
        "kmeans_outlier_quantile": float(kmeans_outlier_quantile),
        "kmeans_inertia": float(kmeans_inertia),
        "kmeans_iterations": int(kmeans_iterations),
        "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(hdbscan_min_samples),
        "hdbscan_metric": str(hdbscan_metric),
        "hdbscan_cluster_selection_method": str(hdbscan_cluster_selection_method),
        "hdbscan_cluster_selection_epsilon": float(hdbscan_cluster_selection_epsilon),
        "hdbscan_target_total_labels": int(hdbscan_target_total_labels),
        "hdbscan_original_normal_clusters": original_normal_clusters,
        "hdbscan_total_labels": int(hdbscan_result["cluster"].nunique()),
        "hdbscan_normal_clusters": int(
            hdbscan_result.loc[
                hdbscan_result["cluster"] != -1, "cluster"
            ].nunique()
        ),
        "hdbscan_noise_rows": noise_rows,
        "hdbscan_noise_percent": 100 * noise_rows / max(len(hdbscan_result), 1),
    }

    return {
        KMEANS_METHOD: kmeans_result,
        HDBSCAN_METHOD: hdbscan_result,
    }, metadata


print("Running initial KMeans and HDBSCAN models...")
DATASETS, LAST_RUN_METADATA = run_algorithms_for_features(DEFAULT_MODEL_FEATURES)
print("Initial models completed.")


# ============================================================
# Dashboard helpers
# ============================================================


def get_dataset(method: str) -> pd.DataFrame:
    return DATASETS[method].copy()


def get_feature_options() -> list[dict[str, str]]:
    """Return all requested features with their native sampling coverage."""
    options: list[dict[str, str]] = []

    for feature in AVAILABLE_FEATURES:
        sampled_percent = 100.0 * RAW_DATA[feature].notna().mean()
        options.append(
            {
                "label": f"{feature} ({sampled_percent:.1f}% sampled)",
                "value": feature,
            }
        )

    return options


def get_job_options() -> list[dict[str, Any]]:
    jobs = sorted(RAW_DATA["job_number"].dropna().astype(int).unique())
    return [{"label": "All jobs", "value": "all"}] + [
        {"label": f"Job {job}", "value": int(job)} for job in jobs
    ]


def get_x_axis_options() -> list[dict[str, str]]:
    options = [{"label": "Row number inside job", "value": "row_in_job"}]

    if "depth" in RAW_DATA.columns:
        options.append({"label": "Depth", "value": "depth"})

    if "time" in RAW_DATA.columns and RAW_DATA["time"].notna().any():
        options.append({"label": "Time", "value": "time"})

    if "ground_depth" in RAW_DATA.columns:
        options.append({"label": "Ground depth", "value": "ground_depth"})

    if (
        "ground_time" in RAW_DATA.columns
        and RAW_DATA["ground_time"].notna().any()
    ):
        options.append({"label": "Ground time", "value": "ground_time"})

    return options


def default_x_axis() -> str:
    values = [option["value"] for option in get_x_axis_options()]

    if "depth" in values:
        return "depth"

    if "time" in values:
        return "time"

    return "row_in_job"


def cluster_label(method: str, cluster_value: int) -> str:
    """Use a readable fixed label for the HDBSCAN noise class."""
    if method == HDBSCAN_METHOD and cluster_value == -1:
        return "Anomaly / Noise"

    return f"Cluster {cluster_value}"


def create_color_mapping(
    method: str,
    cluster_values: list[int],
) -> dict[int, str]:
    """Keep HDBSCAN Anomaly / Noise permanently black."""
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    mapping: dict[int, str] = {}

    normal_clusters = [cluster for cluster in cluster_values if cluster != -1]

    if method == HDBSCAN_METHOD and -1 in cluster_values:
        mapping[-1] = "black"

    for index, cluster in enumerate(normal_clusters):
        mapping[cluster] = palette[index % len(palette)]

    return mapping


def filter_by_job(df: pd.DataFrame, selected_job: Any) -> pd.DataFrame:
    if selected_job == "all":
        return df.copy()

    return df[df["job_number"] == int(selected_job)].copy()


def downsample_by_position(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df

    positions = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[positions].copy()


def method_cluster_options(
    method: str,
    selected_job: Any,
) -> tuple[list[dict[str, Any]], list[int]]:
    df = filter_by_job(get_dataset(method), selected_job)
    cluster_values = sorted(int(value) for value in df["cluster"].dropna().unique())

    options = [
        {
            "label": cluster_label(method, cluster_value),
            "value": cluster_value,
        }
        for cluster_value in cluster_values
    ]

    return options, cluster_values


def make_cluster_summary(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("cluster")
        .agg(rows=("cluster", "size"))
        .reset_index()
    )

    summary["cluster_label"] = summary["cluster"].apply(
        lambda value: cluster_label(method, int(value))
    )
    summary["percent_of_filtered_data"] = 100 * summary["rows"] / len(df)

    if method == KMEANS_METHOD:
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

    if method == HDBSCAN_METHOD:
        summary["cluster_type"] = summary["cluster"].apply(
            lambda value: "anomaly_noise" if int(value) == -1 else "regime"
        )

        if "cluster_probability" in df.columns:
            probability_summary = (
                df.groupby("cluster")["cluster_probability"]
                .mean()
                .reset_index()
                .rename(columns={
                    "cluster_probability": "avg_original_membership_probability"
                })
            )
            summary = summary.merge(probability_summary, on="cluster", how="left")
            summary.loc[
                summary["cluster"] == -1,
                "avg_original_membership_probability",
            ] = np.nan

        if "hdbscan_original_cluster" in df.columns:
            original_summary = (
                df.groupby("cluster")["hdbscan_original_cluster"]
                .apply(
                    lambda values: ", ".join(
                        str(value)
                        for value in sorted(
                            set(int(value) for value in values if pd.notna(value))
                        )
                    )
                )
                .reset_index()
                .rename(columns={
                    "hdbscan_original_cluster": "original_hdbscan_clusters"
                })
            )
            summary = summary.merge(original_summary, on="cluster", how="left")

    front_columns = [
        "cluster",
        "cluster_label",
        "rows",
        "percent_of_filtered_data",
    ]
    other_columns = [
        column for column in summary.columns if column not in front_columns
    ]

    return summary[front_columns + other_columns].sort_values("cluster")


def sanitize_table_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert values to types accepted by Dash DataTable."""
    result = df.copy()

    def sanitize_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            return ", ".join(map(str, value))

        if isinstance(value, dict):
            return str(value)

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        return value

    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)

        result[column] = result[column].apply(sanitize_value)

    return result


def create_stacked_feature_figure(
    df: pd.DataFrame,
    method: str,
    selected_features: list[str],
    x_axis: str,
    selected_clusters: list[int],
    max_points: int,
) -> go.Figure:
    """
    Plot only genuine sampled values.

    Sparse lower-rate channels are neither interpolated nor globally
    downsampled away. Each feature/cluster trace is filtered to rows where
    both the x-axis and feature value are present, then downsampled locally.
    """
    if not selected_features:
        figure = go.Figure()
        figure.update_layout(title="Select at least one feature to plot.")
        return figure

    valid_features = [
        feature for feature in selected_features if feature in df.columns
    ]

    if not valid_features:
        figure = go.Figure()
        figure.update_layout(title="None of the selected plot features is available.")
        return figure

    filtered = df[df["cluster"].isin(selected_clusters)].copy()

    if filtered.empty:
        figure = go.Figure()
        figure.update_layout(title="No data is available for the selected filters.")
        return figure

    if x_axis not in filtered.columns:
        x_axis = "row_in_job"

    cluster_values = sorted(
        int(value) for value in filtered["cluster"].dropna().unique()
    )
    colors = create_color_mapping(method, cluster_values)

    figure = make_subplots(
        rows=len(valid_features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=valid_features,
    )

    for row_index, feature in enumerate(valid_features, start=1):
        for cluster_value in cluster_values:
            cluster_df = filtered.loc[
                filtered["cluster"] == cluster_value,
                ["job_number", x_axis, feature],
            ].copy()

            # Keep only real observations. No interpolation or fill is used.
            cluster_df = cluster_df.dropna(subset=[x_axis, feature])
            cluster_df = cluster_df.sort_values(["job_number", x_axis])

            if cluster_df.empty:
                continue

            cluster_df = downsample_by_position(
                cluster_df,
                max_points=max_points,
            )

            display_label = cluster_label(method, cluster_value)

            figure.add_trace(
                go.Scattergl(
                    x=cluster_df[x_axis],
                    y=cluster_df[feature],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": colors[cluster_value],
                        "opacity": 0.75,
                    },
                    name=display_label,
                    legendgroup=str(cluster_value),
                    showlegend=row_index == 1,
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

        figure.update_yaxes(title_text=feature, row=row_index, col=1)

    figure.update_xaxes(
        title_text=x_axis,
        row=len(valid_features),
        col=1,
    )

    figure.update_layout(
        title=f"{method} clustered time-series view",
        height=max(400, 260 * len(valid_features)),
        template="plotly_white",
        legend_title="Cluster",
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
    )

    return figure


def build_filtered_outputs(
    method: str,
    selected_job: Any,
    x_axis: str,
    selected_features: list[str],
    selected_clusters: list[int] | None,
    max_points: int,
) -> tuple[go.Figure, pd.DataFrame]:
    df = filter_by_job(get_dataset(method), selected_job)

    if not selected_clusters:
        selected_clusters = sorted(
            int(value) for value in df["cluster"].dropna().unique()
        )

    selected_clusters = [int(value) for value in selected_clusters]

    figure = create_stacked_feature_figure(
        df=df,
        method=method,
        selected_features=selected_features or [],
        x_axis=x_axis,
        selected_clusters=selected_clusters,
        max_points=int(max_points or 12000),
    )

    filtered_for_summary = df[df["cluster"].isin(selected_clusters)].copy()
    summary = sanitize_table_dataframe(
        make_cluster_summary(filtered_for_summary, method)
    )

    return figure, summary


def initial_status_text() -> str:
    return (
        f"Initial models completed using: {', '.join(DEFAULT_MODEL_FEATURES)}. "
        f"Modelled {LAST_RUN_METADATA['modelled_rows']:,} of "
        f"{LAST_RUN_METADATA['raw_rows']:,} raw rows using complete cases only; "
        f"no interpolation or imputation. HDBSCAN has "
        f"{LAST_RUN_METADATA['hdbscan_normal_clusters']} regimes plus "
        f"Anomaly / Noise "
        f"({LAST_RUN_METADATA['hdbscan_noise_percent']:.2f}% of modelled rows)."
    )


def numeric_parameter_control(
    label: str,
    component_id: str,
    value: int | float,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    help_text: str = "",
) -> html.Div:
    """Create a compact labelled numeric hyperparameter control."""
    return html.Div(
        children=[
            html.Label(label, style={"fontWeight": "bold"}),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                debounce=True,
                style={"width": "100%", "height": "36px"},
            ),
            html.Small(help_text, style={"display": "block", "marginTop": "4px"}),
        ]
    )


# ============================================================
# Dash application
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "margin": "20px auto",
        "maxWidth": "1900px",
    },
    children=[
        dcc.Store(id="model-version", data=0),
        dcc.Store(id="applied-features-store", data=DEFAULT_MODEL_FEATURES),

        html.H1("Dynamic KMeans vs HDBSCAN Comparison Dashboard"),

        html.P(
            "All requested raw and engineered features use lower-case, "
            "domain-readable names. Datetime streams are shifted by one shared "
            "offset so the earliest timestamp is 2020-01-01. Lower-rate S&V/MWD "
            "measurements are never interpolated or filled. Select input "
            "features, rerun both algorithms on the same complete-case scaled "
            "matrix, and compare the results side by side. HDBSCAN cluster -1 "
            "is always displayed as Anomaly / Noise in black."
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
                    "Input features used to train both algorithms",
                    style={"fontWeight": "bold"},
                ),
                dcc.Dropdown(
                    id="model-feature-dropdown",
                    options=get_feature_options(),
                    value=DEFAULT_MODEL_FEATURES,
                    multi=True,
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "18px",
                        "marginTop": "16px",
                    },
                    children=[
                        html.Div(
                            style={
                                "padding": "12px",
                                "border": "1px solid #dedede",
                                "borderRadius": "8px",
                            },
                            children=[
                                html.H3("KMeans hyperparameters"),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                        "gap": "12px",
                                    },
                                    children=[
                                        numeric_parameter_control(
                                            "Number of clusters (k)",
                                            "kmeans-k-input",
                                            DEFAULT_KMEANS_N_CLUSTERS,
                                            minimum=2, maximum=50, step=1,
                                            help_text="The number of centroid-based regimes.",
                                        ),
                                        numeric_parameter_control(
                                            "Maximum iterations",
                                            "kmeans-max-iter-input",
                                            DEFAULT_KMEANS_MAX_ITER,
                                            minimum=1, maximum=2000, step=10,
                                            help_text="Upper limit for centroid updates.",
                                        ),
                                        numeric_parameter_control(
                                            "Convergence tolerance",
                                            "kmeans-tolerance-input",
                                            DEFAULT_KMEANS_TOLERANCE,
                                            minimum=1e-8, maximum=0.1, step=0.0001,
                                            help_text="Smaller values require tighter convergence.",
                                        ),
                                        numeric_parameter_control(
                                            "Random seed",
                                            "kmeans-random-state-input",
                                            DEFAULT_KMEANS_RANDOM_STATE,
                                            minimum=0, maximum=1000000, step=1,
                                            help_text="Controls reproducible centroid initialisation.",
                                        ),
                                        numeric_parameter_control(
                                            "Outlier quantile",
                                            "kmeans-outlier-quantile-input",
                                            DEFAULT_KMEANS_OUTLIER_QUANTILE,
                                            minimum=0.5, maximum=0.9999, step=0.001,
                                            help_text="Points above this distance quantile are flagged.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "padding": "12px",
                                "border": "1px solid #dedede",
                                "borderRadius": "8px",
                            },
                            children=[
                                html.H3("HDBSCAN hyperparameters"),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                        "gap": "12px",
                                    },
                                    children=[
                                        numeric_parameter_control(
                                            "Minimum cluster size",
                                            "hdbscan-min-cluster-size-input",
                                            DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                                            minimum=2, maximum=100000, step=10,
                                            help_text="Smallest group allowed to form a cluster.",
                                        ),
                                        numeric_parameter_control(
                                            "Minimum samples",
                                            "hdbscan-min-samples-input",
                                            DEFAULT_HDBSCAN_MIN_SAMPLES,
                                            minimum=1, maximum=100000, step=5,
                                            help_text="Higher values are more conservative and create more noise.",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Distance metric", style={"fontWeight": "bold"}),
                                                dcc.Dropdown(
                                                    id="hdbscan-metric-dropdown",
                                                    options=[
                                                        {"label": "Euclidean", "value": "euclidean"},
                                                        {"label": "Manhattan", "value": "manhattan"},
                                                        {"label": "Chebyshev", "value": "chebyshev"},
                                                    ],
                                                    value=DEFAULT_HDBSCAN_METRIC,
                                                    clearable=False,
                                                ),
                                                html.Small("Distance used to measure local density."),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Cluster selection method", style={"fontWeight": "bold"}),
                                                dcc.Dropdown(
                                                    id="hdbscan-selection-method-dropdown",
                                                    options=[
                                                        {"label": "EOM (broader stable clusters)", "value": "eom"},
                                                        {"label": "Leaf (finer clusters)", "value": "leaf"},
                                                    ],
                                                    value=DEFAULT_HDBSCAN_CLUSTER_SELECTION_METHOD,
                                                    clearable=False,
                                                ),
                                                html.Small("Leaf normally produces more fine-grained clusters."),
                                            ]
                                        ),
                                        numeric_parameter_control(
                                            "Cluster selection epsilon",
                                            "hdbscan-epsilon-input",
                                            DEFAULT_HDBSCAN_CLUSTER_SELECTION_EPSILON,
                                            minimum=0.0, maximum=1000.0, step=0.01,
                                            help_text="Merges clusters separated by less than this distance.",
                                        ),
                                        numeric_parameter_control(
                                            "Target total displayed labels",
                                            "hdbscan-target-total-labels-input",
                                            DEFAULT_HDBSCAN_TARGET_TOTAL_LABELS,
                                            minimum=2, maximum=50, step=1,
                                            help_text="Includes the fixed Anomaly / Noise label -1; this is post-processing.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Button(
                    "Apply features and rerun both algorithms",
                    id="run-models-button",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "height": "40px",
                        "fontWeight": "bold",
                    },
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="model-run-status",
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
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "18px",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Job number"),
                        dcc.Dropdown(
                            id="job-dropdown",
                            options=get_job_options(),
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
                            options=get_x_axis_options(),
                            value=default_x_axis(),
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Max points per chart"),
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
                        html.Label("Features to plot on both charts"),
                        dcc.Dropdown(
                            id="plot-feature-dropdown",
                            options=get_feature_options(),
                            value=DEFAULT_VISIBLE_FEATURES,
                            multi=True,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("KMeans clusters to display"),
                        dcc.Dropdown(
                            id="kmeans-cluster-dropdown",
                            multi=True,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("HDBSCAN labels to display"),
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
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "alignItems": "start",
            },
            children=[
                dcc.Graph(id="kmeans-graph", style={"width": "100%"}),
                dcc.Graph(id="hdbscan-graph", style={"width": "100%"}),
            ],
        ),

        html.H2("Cluster summaries for the current selection"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "alignItems": "start",
            },
            children=[
                html.Div(
                    children=[
                        html.H3("KMeans summary"),
                        dash_table.DataTable(
                            id="kmeans-summary-table",
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
                    ]
                ),
                html.Div(
                    children=[
                        html.H3("HDBSCAN summary"),
                        dash_table.DataTable(
                            id="hdbscan-summary-table",
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
                            style_data_conditional=cast(
                                Any,
                                [
                                    {
                                        "if": {
                                            "filter_query": "{cluster} = -1",
                                        },
                                        "backgroundColor": "black",
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
    Output("model-run-status", "children"),
    Input("run-models-button", "n_clicks"),
    State("model-feature-dropdown", "value"),
    State("kmeans-k-input", "value"),
    State("kmeans-max-iter-input", "value"),
    State("kmeans-tolerance-input", "value"),
    State("kmeans-random-state-input", "value"),
    State("kmeans-outlier-quantile-input", "value"),
    State("hdbscan-min-cluster-size-input", "value"),
    State("hdbscan-min-samples-input", "value"),
    State("hdbscan-metric-dropdown", "value"),
    State("hdbscan-selection-method-dropdown", "value"),
    State("hdbscan-epsilon-input", "value"),
    State("hdbscan-target-total-labels-input", "value"),
    prevent_initial_call=True,
)
def rerun_models_with_selected_features(
    n_clicks: int,
    selected_features: list[str] | None,
    kmeans_n_clusters: int | None,
    kmeans_max_iter: int | None,
    kmeans_tolerance: float | None,
    kmeans_random_state: int | None,
    kmeans_outlier_quantile: float | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    hdbscan_metric: str | None,
    hdbscan_selection_method: str | None,
    hdbscan_epsilon: float | None,
    hdbscan_target_total_labels: int | None,
):
    del n_clicks

    if not selected_features:
        return no_update, no_update, "Select at least one input feature."

    try:
        updated_datasets, metadata = run_algorithms_for_features(
            selected_features,
            kmeans_n_clusters=int(kmeans_n_clusters or DEFAULT_KMEANS_N_CLUSTERS),
            kmeans_max_iter=int(kmeans_max_iter or DEFAULT_KMEANS_MAX_ITER),
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
                hdbscan_min_cluster_size or DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
            ),
            hdbscan_min_samples=int(
                hdbscan_min_samples or DEFAULT_HDBSCAN_MIN_SAMPLES
            ),
            hdbscan_metric=hdbscan_metric or DEFAULT_HDBSCAN_METRIC,
            hdbscan_cluster_selection_method=(
                hdbscan_selection_method
                or DEFAULT_HDBSCAN_CLUSTER_SELECTION_METHOD
            ),
            hdbscan_cluster_selection_epsilon=float(
                hdbscan_epsilon
                if hdbscan_epsilon is not None
                else DEFAULT_HDBSCAN_CLUSTER_SELECTION_EPSILON
            ),
            hdbscan_target_total_labels=int(
                hdbscan_target_total_labels
                or DEFAULT_HDBSCAN_TARGET_TOTAL_LABELS
            ),
        )

        DATASETS.clear()
        DATASETS.update(updated_datasets)

        LAST_RUN_METADATA.clear()
        LAST_RUN_METADATA.update(metadata)

        model_version = int(time.time() * 1000)
        status = (
            f"Models rerun using: {', '.join(selected_features)}. "
            f"Modelled {metadata['modelled_rows']:,} of "
            f"{metadata['raw_rows']:,} raw rows; "
            f"{metadata['excluded_rows']:,} rows were excluded because at least "
            f"one selected feature was genuinely missing. No interpolation or "
            f"imputation was used. Completed in "
            f"{metadata['elapsed_seconds']:.2f} seconds. "
            f"KMeans: k={metadata['kmeans_n_clusters']}, "
            f"iterations={metadata['kmeans_iterations']}, "
            f"inertia={metadata['kmeans_inertia']:.2f}. "
            f"HDBSCAN native result: "
            f"{metadata['hdbscan_original_normal_clusters']} normal clusters; "
            f"displayed result: {metadata['hdbscan_normal_clusters']} regimes plus "
            f"Anomaly / Noise ({metadata['hdbscan_noise_rows']:,} rows; "
            f"{metadata['hdbscan_noise_percent']:.2f}%)."
        )

        return model_version, selected_features, status

    except Exception as error:  # Display a useful error in the app.
        return no_update, no_update, f"Model run failed: {error}"


@app.callback(
    Output("kmeans-cluster-dropdown", "options"),
    Output("kmeans-cluster-dropdown", "value"),
    Output("hdbscan-cluster-dropdown", "options"),
    Output("hdbscan-cluster-dropdown", "value"),
    Input("model-version", "data"),
    Input("job-dropdown", "value"),
)
def update_cluster_options(model_version: int, selected_job: Any):
    del model_version

    kmeans_options, kmeans_values = method_cluster_options(
        KMEANS_METHOD, selected_job
    )
    hdbscan_options, hdbscan_values = method_cluster_options(
        HDBSCAN_METHOD, selected_job
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
    Input("job-dropdown", "value"),
    Input("x-axis-dropdown", "value"),
    Input("plot-feature-dropdown", "value"),
    Input("kmeans-cluster-dropdown", "value"),
    Input("hdbscan-cluster-dropdown", "value"),
    Input("max-points-input", "value"),
)
def update_comparison_view(
    model_version: int,
    selected_job: Any,
    x_axis: str,
    selected_features: list[str] | None,
    selected_kmeans_clusters: list[int] | None,
    selected_hdbscan_clusters: list[int] | None,
    max_points: int | None,
):
    del model_version

    kmeans_figure, kmeans_summary = build_filtered_outputs(
        method=KMEANS_METHOD,
        selected_job=selected_job,
        x_axis=x_axis,
        selected_features=selected_features or [],
        selected_clusters=selected_kmeans_clusters,
        max_points=int(max_points or 12000),
    )

    hdbscan_figure, hdbscan_summary = build_filtered_outputs(
        method=HDBSCAN_METHOD,
        selected_job=selected_job,
        x_axis=x_axis,
        selected_features=selected_features or [],
        selected_clusters=selected_hdbscan_clusters,
        max_points=int(max_points or 12000),
    )

    kmeans_columns = [
        {"name": column, "id": column} for column in kmeans_summary.columns
    ]
    hdbscan_columns = [
        {"name": column, "id": column} for column in hdbscan_summary.columns
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
