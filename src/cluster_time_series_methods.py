from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "time_series_jobs"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
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

# KMeans returns 7 operational clusters.
KMEANS_N_CLUSTERS = 7

# HDBSCAN may return more than 7 non-noise clusters.
# This value merges only the non-noise HDBSCAN clusters down to 7.
# Noise/outlier cluster -1 is kept untouched.
#
# If you want 7 total labels INCLUDING noise, change this to 6.
HDBSCAN_TARGET_NORMAL_CLUSTERS = 6

KMEANS_RANDOM_STATE = 43
KMEANS_OUTLIER_QUANTILE = 0.98

HDBSCAN_MIN_CLUSTER_SIZE = 300
HDBSCAN_MIN_SAMPLES = 50
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"


# ============================================================
# Data loading
# ============================================================

def extract_job_number_from_source_file(source_file: str) -> int:
    """
    Extract job number from file names such as:

    time_series_job1.csv
    time_series_job2.csv
    """
    match = re.search(r"job(\d+)", str(source_file).lower())

    if match:
        return int(match.group(1))

    return -1


def load_all_csv_files(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files from the raw data folder and combine them into one DataFrame.
    Each row keeps the source file name and job number.
    """
    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {raw_data_dir}")

    frames = []

    for file_path in csv_files:
        print(f"Loading: {file_path.name}")

        temp = pd.read_csv(file_path)
        temp["source_file"] = file_path.name
        temp["job_number"] = extract_job_number_from_source_file(file_path.name)

        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)

    print("\nCombined data shape:", combined.shape)
    print("Files loaded:", len(csv_files))

    return combined


# ============================================================
# Feature engineering helpers
# ============================================================

def circular_difference_degrees(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Calculate circular difference between two angle columns in degrees.

    Example:
    359 degrees and 1 degree are only 2 degrees apart,
    not 358 degrees apart.
    """
    return ((a - b + 180) % 360) - 180


def add_angle_sin_cos_features(data: pd.DataFrame, angle_col: str) -> pd.DataFrame:
    """
    Convert an angle column into sine and cosine features.
    """
    radians = np.deg2rad(data[angle_col])

    data[f"{angle_col}_sin"] = np.sin(radians)
    data[f"{angle_col}_cos"] = np.cos(radians)

    return data


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Convert raw time-series/depth data into clustering features.

    This function keeps the same feature set for both KMeans and HDBSCAN
    so the method comparison is fair.
    """
    data = df.copy()

    if "date_time_pd" in data.columns:
        data["date_time_pd"] = pd.to_datetime(data["date_time_pd"], errors="coerce")

    sort_columns = ["source_file"]

    if "MD" in data.columns:
        sort_columns.append("MD")
    elif "date_time_pd" in data.columns:
        sort_columns.append("date_time_pd")

    data = data.sort_values(sort_columns).reset_index(drop=True)

    if "MD" in data.columns:
        data["depth_step"] = data.groupby("source_file")["MD"].diff()

    if "date_time_pd" in data.columns:
        data["time_step_sec"] = (
            data.groupby("source_file")["date_time_pd"]
            .diff()
            .dt.total_seconds()
        )

    if "depth_step" in data.columns and "time_step_sec" in data.columns:
        data["rop_m_per_hr"] = data["depth_step"] / (data["time_step_sec"] / 3600.0)
        data.loc[~np.isfinite(data["rop_m_per_hr"]), "rop_m_per_hr"] = np.nan

    for angle_col in ["CT_AZI", "mwd_azim"]:
        if angle_col in data.columns:
            data = add_angle_sin_cos_features(data, angle_col)

    if "CT_INC" in data.columns and "mwd_incl" in data.columns:
        data["incl_diff"] = data["CT_INC"] - data["mwd_incl"]

    if "CT_AZI" in data.columns and "mwd_azim" in data.columns:
        data["azim_diff"] = circular_difference_degrees(data["CT_AZI"], data["mwd_azim"])

    if "RPM_UT" in data.columns and "RPM_LT" in data.columns:
        data["rpm_diff_ut_lt"] = data["RPM_UT"] - data["RPM_LT"]
        data["rpm_mean_lt_ut"] = (data["RPM_UT"] + data["RPM_LT"]) / 2.0

    if {"GX", "GY", "GZ"}.issubset(data.columns):
        data["g_magnitude"] = np.sqrt(data["GX"] ** 2 + data["GY"] ** 2 + data["GZ"] ** 2)

    if {"BX", "BY", "BZ"}.issubset(data.columns):
        data["b_magnitude"] = np.sqrt(data["BX"] ** 2 + data["BY"] ** 2 + data["BZ"] ** 2)

    feature_cols = [col for col in FEATURES_USED_FOR_CLUSTERING if col in data.columns]

    if len(feature_cols) == 0:
        raise ValueError("No usable numerical features were found.")

    missing_features = sorted(set(FEATURES_USED_FOR_CLUSTERING) - set(feature_cols))
    if missing_features:
        print("\nWarning: these configured features were not found and will be skipped:")
        for col in missing_features:
            print(f" - {col}")

    X_df = data[feature_cols].copy()

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    for col in X_df.columns:
        median_value = X_df[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X_df[col] = X_df[col].fillna(median_value)

    print("\nSelected features:")
    for col in feature_cols:
        print(f" - {col}")

    print("\nFeature matrix shape:", X_df.shape)

    return data, X_df, feature_cols


# ============================================================
# Preprocessing
# ============================================================

def robust_clip(X_df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Clip extreme values so very large spikes do not dominate distance-based clustering.
    """
    lower = X_df.quantile(lower_q)
    upper = X_df.quantile(upper_q)

    return X_df.clip(lower=lower, upper=upper, axis=1)


def standardize(X_df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Standardize features:

        z = (x - mean) / standard deviation

    KMeans and HDBSCAN both use distance-based logic, so scaling is essential.
    """
    mean_values = X_df.mean()
    std_values = X_df.std(ddof=0).replace(0, 1)

    X_scaled = (X_df - mean_values) / std_values

    return X_scaled.to_numpy(dtype=float), mean_values, std_values


# ============================================================
# KMeans from scratch using NumPy
# ============================================================

def kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    KMeans++ initialization.
    """
    rng = np.random.default_rng(random_state)

    n_rows, n_features = X.shape
    centroids = np.empty((k, n_features), dtype=float)

    first_index = rng.integers(0, n_rows)
    centroids[0] = X[first_index]

    closest_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

    for c in range(1, k):
        probabilities = closest_dist_sq / closest_dist_sq.sum()

        next_index = rng.choice(n_rows, p=probabilities)
        centroids[c] = X[next_index]

        new_dist_sq = np.sum((X - centroids[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centroids


def assign_clusters(X: np.ndarray, centroids: np.ndarray, chunk_size: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each row to the nearest centroid.
    """
    n_rows = X.shape[0]

    labels = np.empty(n_rows, dtype=int)
    min_distances_sq = np.empty(n_rows, dtype=float)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        distances_sq = ((X[start:end, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

        labels[start:end] = np.argmin(distances_sq, axis=1)
        min_distances_sq[start:end] = np.min(distances_sq, axis=1)

    return labels, min_distances_sq


def kmeans_numpy(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    """
    Train KMeans using only NumPy.
    """
    centroids = kmeans_plus_plus_init(X, k=k, random_state=random_state)

    previous_inertia: Optional[float] = None

    for iteration in range(max_iter):
        labels, min_distances_sq = assign_clusters(X, centroids)
        inertia = float(min_distances_sq.sum())

        new_centroids = np.empty_like(centroids)

        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                farthest_index = np.argmax(min_distances_sq)
                new_centroids[cluster_id] = X[farthest_index]
            else:
                new_centroids[cluster_id] = cluster_points.mean(axis=0)

        centroid_shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids

        if previous_inertia is not None:
            relative_improvement = (previous_inertia - inertia) / max(previous_inertia, 1e-12)

            if centroid_shift < tol or abs(relative_improvement) < tol:
                break

        previous_inertia = inertia

    final_labels, final_distances_sq = assign_clusters(X, centroids)
    final_inertia = float(final_distances_sq.sum())

    return final_labels, centroids, final_inertia, iteration + 1, final_distances_sq


# ============================================================
# KMeans pipeline
# ============================================================

def create_kmeans_summary(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Create a KMeans cluster summary table.
    """
    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
            possible_outliers=("possible_outlier", "sum"),
            avg_cluster_distance=("cluster_distance", "mean"),
        )
        .reset_index()
    )

    summary["method"] = "KMeans"
    summary["cluster_type"] = "cluster"
    summary["percent_of_data"] = 100 * summary["rows"] / len(data)

    feature_means = data.groupby("cluster")[feature_cols].mean().reset_index()
    summary = summary.merge(feature_means, on="cluster", how="left")

    front_cols = [
        "method",
        "cluster",
        "cluster_type",
        "rows",
        "percent_of_data",
        "possible_outliers",
        "avg_cluster_distance",
    ]

    other_cols = [col for col in summary.columns if col not in front_cols]
    summary = summary[front_cols + other_cols]

    return summary


def run_kmeans_pipeline(
    data: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run the KMeans baseline with exactly 7 clusters.
    """
    print(f"\nTraining KMeans baseline with k={KMEANS_N_CLUSTERS}...")

    labels, centroids, inertia, n_iter, dist_sq = kmeans_numpy(
        X_scaled,
        k=KMEANS_N_CLUSTERS,
        max_iter=100,
        tol=1e-4,
        random_state=KMEANS_RANDOM_STATE,
    )

    result = data.copy()
    result["method"] = "KMeans"
    result["cluster"] = labels
    result["kmeans_cluster"] = labels
    result["cluster_distance"] = np.sqrt(dist_sq)

    outlier_threshold = result["cluster_distance"].quantile(KMEANS_OUTLIER_QUANTILE)
    result["possible_outlier"] = result["cluster_distance"] >= outlier_threshold

    summary = create_kmeans_summary(result, feature_cols)

    output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs_kmeans.csv"
    legacy_output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs.csv"
    summary_path = OUTPUTS_DIR / "kmeans_cluster_summary.csv"
    legacy_summary_path = OUTPUTS_DIR / "cluster_summary.csv"

    result.to_csv(output_path, index=False)
    result.to_csv(legacy_output_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(legacy_summary_path, index=False)

    print("KMeans clustered data saved to:", output_path)
    print("KMeans cluster summary saved to:", summary_path)
    print("KMeans inertia:", round(inertia, 2))
    print("KMeans iterations:", n_iter)
    print("KMeans possible outlier threshold:", round(float(outlier_threshold), 4))

    return result


# ============================================================
# HDBSCAN pipeline
# ============================================================

def run_hdbscan_model(X_scaled: np.ndarray) -> HDBSCAN:
    """
    Run HDBSCAN clustering.

    HDBSCAN can return more than 7 clusters and can also label noise as -1.
    """
    model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    return model


def merge_hdbscan_clusters_to_target(
    labels: np.ndarray,
    X_scaled: np.ndarray,
    target_normal_clusters: int = HDBSCAN_TARGET_NORMAL_CLUSTERS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Merge close HDBSCAN non-noise clusters until there are target_normal_clusters.

    Important:
    - HDBSCAN noise/outlier cluster -1 is kept untouched.
    - Only normal clusters are merged.
    - Merging is based on distances between original HDBSCAN cluster centroids
      in the same scaled feature space used for clustering.
    """
    labels = np.asarray(labels)
    normal_labels = sorted(int(label) for label in np.unique(labels) if int(label) != -1)

    merged_labels = np.full(labels.shape, fill_value=-1, dtype=int)

    if len(normal_labels) == 0:
        merge_map = pd.DataFrame(
            columns=[
                "hdbscan_original_cluster",
                "hdbscan_merged_cluster",
                "rows",
                "note",
            ]
        )
        return merged_labels, merge_map

    original_cluster_rows = {
        label: int(np.sum(labels == label))
        for label in normal_labels
    }

    if len(normal_labels) <= target_normal_clusters:
        print(
            f"\nHDBSCAN returned {len(normal_labels)} non-noise clusters, "
            f"which is not more than the target {target_normal_clusters}."
        )
        print("No cluster merging is needed. Clusters will only be relabelled to 0..n-1.")

        original_to_merged = {
            original_label: new_label
            for new_label, original_label in enumerate(normal_labels)
        }
        note = "not_merged_relabelled_only"

    else:
        print(
            f"\nHDBSCAN returned {len(normal_labels)} non-noise clusters."
        )
        print(
            f"Merging close non-noise clusters down to {target_normal_clusters} "
            "while keeping noise cluster -1 untouched..."
        )

        centroids = []

        for label in normal_labels:
            cluster_points = X_scaled[labels == label]
            centroids.append(cluster_points.mean(axis=0))

        centroids = np.vstack(centroids)

        merger = AgglomerativeClustering(
            n_clusters=target_normal_clusters,
            linkage="ward",
        )

        merged_group_ids = merger.fit_predict(centroids)

        # Make merged labels stable/readable by ordering groups using
        # the smallest original HDBSCAN cluster id inside each merged group.
        group_to_originals: dict[int, list[int]] = {}

        for original_label, group_id in zip(normal_labels, merged_group_ids):
            group_to_originals.setdefault(int(group_id), []).append(int(original_label))

        ordered_groups = sorted(
            group_to_originals.keys(),
            key=lambda group_id: min(group_to_originals[group_id]),
        )

        group_to_final_label = {
            group_id: final_label
            for final_label, group_id in enumerate(ordered_groups)
        }

        original_to_merged = {
            original_label: group_to_final_label[int(group_id)]
            for original_label, group_id in zip(normal_labels, merged_group_ids)
        }

        note = "merged_by_agglomerative_centroid_distance"

    for original_label, merged_label in original_to_merged.items():
        merged_labels[labels == original_label] = merged_label

    merge_map = pd.DataFrame(
        [
            {
                "hdbscan_original_cluster": original_label,
                "hdbscan_merged_cluster": merged_label,
                "rows": original_cluster_rows[original_label],
                "note": note,
            }
            for original_label, merged_label in sorted(original_to_merged.items())
        ]
    )

    return merged_labels, merge_map


def create_hdbscan_merged_summary(data: pd.DataFrame, merge_map: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary for merged HDBSCAN clusters.

    cluster = -1 means HDBSCAN noise/outlier and is kept untouched.
    """
    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
            avg_cluster_probability=("cluster_probability", "mean"),
            original_hdbscan_clusters=("hdbscan_original_cluster", lambda x: sorted(set(int(v) for v in x))),
        )
        .reset_index()
    )

    summary["method"] = "HDBSCAN_merged_to_7"
    summary["cluster_type"] = np.where(
        summary["cluster"] == -1,
        "noise_outlier",
        "merged_cluster",
    )
    summary["percent_of_data"] = 100 * summary["rows"] / len(data)

    # Probability for noise is not a true cluster-membership probability.
    summary.loc[summary["cluster"] == -1, "avg_cluster_probability"] = np.nan

    front_cols = [
        "method",
        "cluster",
        "cluster_type",
        "rows",
        "percent_of_data",
        "avg_cluster_probability",
        "original_hdbscan_clusters",
    ]

    summary = summary[front_cols]

    return summary.sort_values("cluster").reset_index(drop=True)


def run_hdbscan_pipeline(data: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Run HDBSCAN, then merge non-noise HDBSCAN clusters to 7.
    """
    print("\nRunning HDBSCAN...")
    print("This may take a little time on 50,000+ rows.")

    model = run_hdbscan_model(X_scaled)

    original_labels = model.labels_

    merged_labels, merge_map = merge_hdbscan_clusters_to_target(
        labels=original_labels,
        X_scaled=X_scaled,
        target_normal_clusters=HDBSCAN_TARGET_NORMAL_CLUSTERS,
    )

    result = data.copy()
    result["method"] = "HDBSCAN_merged_to_7"
    result["hdbscan_original_cluster"] = original_labels
    result["cluster"] = merged_labels
    result["hdbscan_merged_cluster"] = merged_labels
    result["hdbscan_noise_outlier"] = original_labels == -1

    if hasattr(model, "probabilities_"):
        result["cluster_probability"] = model.probabilities_
    else:
        result["cluster_probability"] = np.nan

    summary = create_hdbscan_merged_summary(result, merge_map)

    output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs_hdbscan_merged.csv"
    legacy_output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs_hdbscan.csv"
    summary_path = OUTPUTS_DIR / "hdbscan_merged_cluster_summary.csv"
    legacy_summary_path = OUTPUTS_DIR / "hdbscan_cluster_summary.csv"
    merge_map_path = OUTPUTS_DIR / "hdbscan_cluster_merge_map.csv"

    result.to_csv(output_path, index=False)
    result.to_csv(legacy_output_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(legacy_summary_path, index=False)
    merge_map.to_csv(merge_map_path, index=False)

    print("HDBSCAN merged clustered data saved to:", output_path)
    print("HDBSCAN merged cluster summary saved to:", summary_path)
    print("HDBSCAN cluster merge map saved to:", merge_map_path)

    original_normal_clusters = result.loc[
        result["hdbscan_original_cluster"] != -1,
        "hdbscan_original_cluster",
    ].nunique()

    merged_normal_clusters = result.loc[
        result["cluster"] != -1,
        "cluster",
    ].nunique()

    n_noise = int(result["hdbscan_noise_outlier"].sum())

    print("\nOriginal HDBSCAN normal clusters:", original_normal_clusters)
    print("Merged HDBSCAN normal clusters:", merged_normal_clusters)
    print("Noise/outlier rows kept untouched:", n_noise)
    print("Noise/outlier percentage:", round(100 * n_noise / len(result), 2), "%")

    return result


# ============================================================
# Method comparison
# ============================================================

def create_method_comparison_summary(kmeans_data: pd.DataFrame, hdbscan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level comparison summary for KMeans and merged HDBSCAN.
    """
    rows = []

    rows.append(
        {
            "method": "KMeans",
            "normal_clusters": int(kmeans_data["cluster"].nunique()),
            "noise_cluster_kept": False,
            "anomaly_definition": f"cluster_distance >= {KMEANS_OUTLIER_QUANTILE:.2f} quantile",
            "anomaly_rows": int(kmeans_data["possible_outlier"].sum()),
            "anomaly_percent": 100 * float(kmeans_data["possible_outlier"].mean()),
        }
    )

    rows.append(
        {
            "method": "HDBSCAN_merged_to_7",
            "normal_clusters": int(hdbscan_data.loc[hdbscan_data["cluster"] != -1, "cluster"].nunique()),
            "noise_cluster_kept": True,
            "anomaly_definition": "original HDBSCAN label == -1",
            "anomaly_rows": int(hdbscan_data["hdbscan_noise_outlier"].sum()),
            "anomaly_percent": 100 * float(hdbscan_data["hdbscan_noise_outlier"].mean()),
        }
    )

    comparison = pd.DataFrame(rows)
    comparison_path = OUTPUTS_DIR / "method_comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nMethod comparison summary saved to:", comparison_path)
    print(comparison)

    return comparison


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("Starting combined clustering pipeline...")
    print("Raw data folder:", RAW_DATA_DIR)

    raw_df = load_all_csv_files(RAW_DATA_DIR)
    data, X_df, feature_cols = build_features(raw_df)

    X_clipped = robust_clip(X_df)
    X_scaled, mean_values, std_values = standardize(X_clipped)

    # Save preprocessing statistics for reproducibility.
    preprocessing_stats = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean": [mean_values[col] for col in feature_cols],
            "std": [std_values[col] for col in feature_cols],
        }
    )
    preprocessing_stats_path = OUTPUTS_DIR / "preprocessing_scaler_stats.csv"
    preprocessing_stats.to_csv(preprocessing_stats_path, index=False)

    kmeans_data = run_kmeans_pipeline(
        data=data,
        X_scaled=X_scaled,
        feature_cols=feature_cols,
    )

    hdbscan_data = run_hdbscan_pipeline(
        data=data,
        X_scaled=X_scaled,
    )

    create_method_comparison_summary(
        kmeans_data=kmeans_data,
        hdbscan_data=hdbscan_data,
    )

    print("\nDone.")
    print("Preprocessing stats saved to:", preprocessing_stats_path)
    print("\nRecommended next command:")
    print("python src/plot_kmeans_hdbscan_time_series.py")


if __name__ == "__main__":
    main()
