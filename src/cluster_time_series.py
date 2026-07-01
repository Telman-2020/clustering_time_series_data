from pathlib import Path
import re
import numpy as np
import pandas as pd


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
# Step 1: Load all CSV files
# ============================================================

def load_all_csv_files(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files from the raw data folder and combine them into one DataFrame.
    Each row keeps the source file name so we know which job/run it came from.
    """
    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {raw_data_dir}")

    frames = []

    for file_path in csv_files:
        print(f"Loading: {file_path.name}")

        temp = pd.read_csv(file_path)
        temp["source_file"] = file_path.name

        job_match = re.search(r"job(\d+)", file_path.stem.lower())
        temp["job_number"] = int(job_match.group(1)) if job_match else -1

        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)

    print("\nCombined data shape:", combined.shape)
    print("Files loaded:", len(csv_files))

    return combined


# ============================================================
# Step 2: Helper functions for feature engineering
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

    This is important because angles are circular.
    For example, 359 degrees and 1 degree are close to each other.
    """
    radians = np.deg2rad(data[angle_col])

    data[f"{angle_col}_sin"] = np.sin(radians)
    data[f"{angle_col}_cos"] = np.cos(radians)

    return data


# ============================================================
# Step 3: Build modelling features
# ============================================================

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Convert raw time-series/depth data into clustering features.

    Important:
    KMeans does not understand time-series directly.
    We create numerical behaviour features first.
    """
    data = df.copy()

    # Convert datetime column if it exists.
    if "date_time_pd" in data.columns:
        data["date_time_pd"] = pd.to_datetime(data["date_time_pd"], errors="coerce")

    # Sort each run/job by measured depth if MD exists.
    if "MD" in data.columns:
        data = data.sort_values(["source_file", "MD"]).reset_index(drop=True)

    # Depth step: how much the measured depth changed from previous row.
    if "MD" in data.columns:
        data["depth_step"] = data.groupby("source_file")["MD"].diff()

    # Time step: seconds between records.
    if "date_time_pd" in data.columns:
        data["time_step_sec"] = (
            data.groupby("source_file")["date_time_pd"]
            .diff()
            .dt.total_seconds()
        )

    # ROP-style feature: depth change per hour.
    if "depth_step" in data.columns and "time_step_sec" in data.columns:
        data["rop_m_per_hr"] = data["depth_step"] / (data["time_step_sec"] / 3600.0)
        data.loc[~np.isfinite(data["rop_m_per_hr"]), "rop_m_per_hr"] = np.nan

    # Convert angle columns to sine/cosine.
    for angle_col in ["CT_AZI", "mwd_azim"]:
        if angle_col in data.columns:
            data = add_angle_sin_cos_features(data, angle_col)

    # Difference between continuous tool and MWD measurements.
    if "CT_INC" in data.columns and "mwd_incl" in data.columns:
        data["incl_diff"] = data["CT_INC"] - data["mwd_incl"]

    if "CT_AZI" in data.columns and "mwd_azim" in data.columns:
        data["azim_diff"] = circular_difference_degrees(data["CT_AZI"], data["mwd_azim"])

    # RPM features.
    if "RPM_UT" in data.columns and "RPM_LT" in data.columns:
        data["rpm_diff_ut_lt"] = data["RPM_UT"] - data["RPM_LT"]
        data["rpm_mean_lt_ut"] = (data["RPM_UT"] + data["RPM_LT"]) / 2.0

    # Vector magnitude features.
    if {"GX", "GY", "GZ"}.issubset(data.columns):
        data["g_magnitude"] = np.sqrt(data["GX"] ** 2 + data["GY"] ** 2 + data["GZ"] ** 2)

    if {"BX", "BY", "BZ"}.issubset(data.columns):
        data["b_magnitude"] = np.sqrt(data["BX"] ** 2 + data["BY"] ** 2 + data["BZ"] ** 2)

    # Candidate feature list.
    candidate_features = [         
       
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

    # Use only features that really exist in the data.
    feature_cols = [col for col in candidate_features if col in data.columns]

    if len(feature_cols) == 0:
        raise ValueError("No usable numerical features were found.")

    X_df = data[feature_cols].copy()

    # Convert everything to numeric.
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Fill missing values using median.
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
# Step 4: Scale features
# ============================================================

def robust_clip(X_df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Clip extreme values so very large outliers do not dominate clustering.
    """
    lower = X_df.quantile(lower_q)
    upper = X_df.quantile(upper_q)

    return X_df.clip(lower=lower, upper=upper, axis=1)


def standardize(X_df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Standardize features using:

        z = (x - mean) / standard deviation

    This is necessary because KMeans is distance-based.
    """
    mean_values = X_df.mean()
    std_values = X_df.std(ddof=0).replace(0, 1)

    X_scaled = (X_df - mean_values) / std_values

    return X_scaled.to_numpy(dtype=float), mean_values, std_values


# ============================================================
# Step 5: KMeans from scratch using NumPy
# ============================================================

def kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    KMeans++ initialization.

    This selects better initial centroids than simple random selection.
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

    Chunking avoids creating a very large distance matrix at once.
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

    Returns:
    - labels
    - centroids
    - inertia
    - number of iterations
    - squared distance of each point to its cluster centroid
    """
    centroids = kmeans_plus_plus_init(X, k=k, random_state=random_state)

    previous_inertia = None

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
# Step 6: Summarise clusters
# ============================================================

def create_cluster_summary(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Create a simple cluster summary table.
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

    summary["percent_of_data"] = 100 * summary["rows"] / len(data)

    # Add feature means per cluster for interpretation.
    feature_means = data.groupby("cluster")[feature_cols].mean().reset_index()

    summary = summary.merge(feature_means, on="cluster", how="left")

    return summary


# ============================================================
# Step 7: Main execution
# ============================================================

def main() -> None:
    print("Starting clustering pipeline...")
    print("Raw data folder:", RAW_DATA_DIR)

    df = load_all_csv_files(RAW_DATA_DIR)

    data, X_df, feature_cols = build_features(df)

    X_clipped = robust_clip(X_df)
    X, mean_values, std_values = standardize(X_clipped)

    # Try several values of k.
    print("\nTesting different numbers of clusters:")

    k_results = []

    for k in range(2, 9):
        labels, centroids, inertia, n_iter, dist_sq = kmeans_numpy(
            X,
            k=k,
            max_iter=100,
            tol=1e-4,
            random_state=42,
        )

        k_results.append(
            {
                "k": k,
                "inertia": inertia,
                "iterations": n_iter,
            }
        )

        print(f"k={k}, inertia={inertia:.2f}, iterations={n_iter}")

    k_results_df = pd.DataFrame(k_results)
    k_results_path = OUTPUTS_DIR / "kmeans_k_selection_results.csv"
    k_results_df.to_csv(k_results_path, index=False)

    # Choose a first practical number of clusters.
    chosen_k = 7

    print(f"\nTraining final KMeans model with k={chosen_k}...")

    labels, centroids, inertia, n_iter, dist_sq = kmeans_numpy(
        X,
        k=chosen_k,
        max_iter=100,
        tol=1e-4,
        random_state=43,
    )

    data["cluster"] = labels
    data["cluster_distance"] = np.sqrt(dist_sq)

    # Simple possible outlier flag:
    # top 2% farthest points from their assigned centroid.
    outlier_threshold = data["cluster_distance"].quantile(0.98)
    data["possible_outlier"] = data["cluster_distance"] >= outlier_threshold

    cluster_summary = create_cluster_summary(data, feature_cols)

    clustered_output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs.csv"
    summary_output_path = OUTPUTS_DIR / "cluster_summary.csv"

    data.to_csv(clustered_output_path, index=False)
    cluster_summary.to_csv(summary_output_path, index=False)

    print("\nDone.")
    print("Clustered data saved to:", clustered_output_path)
    print("Cluster summary saved to:", summary_output_path)
    print("K selection results saved to:", k_results_path)

    print("\nCluster size summary:")
    print(cluster_summary[["cluster", "rows", "percent_of_data", "possible_outliers"]])


if __name__ == "__main__":
    main()
