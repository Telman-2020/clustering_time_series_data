from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


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
# Step 1: Load data
# ============================================================

def extract_job_number_from_source_file(source_file: str) -> int:
    """
    Extract job number from file names like:

    time_series_job1.csv
    time_series_job2.csv
    """
    match = re.search(r"job(\d+)", str(source_file).lower())

    if match:
        return int(match.group(1))

    return -1


def load_all_csv_files(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files and stack them vertically into one DataFrame.
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
# Step 2: Prepare feature matrix
# ============================================================

def prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select only the chosen clustering features and prepare them for HDBSCAN.
    """
    data = df.copy()

    missing_features = [
        col for col in FEATURES_USED_FOR_CLUSTERING
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    X_df = data[FEATURES_USED_FOR_CLUSTERING].copy()

    # Convert all features to numeric.
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Fill missing values using median.
    for col in X_df.columns:
        median_value = X_df[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X_df[col] = X_df[col].fillna(median_value)

    print("\nSelected HDBSCAN features:")
    for col in FEATURES_USED_FOR_CLUSTERING:
        print(f" - {col}")

    print("\nFeature matrix shape:", X_df.shape)

    return data, X_df


def robust_clip(X_df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Clip extreme values before scaling.

    This reduces the effect of very large spikes on distance-based clustering.
    """
    lower = X_df.quantile(lower_q)
    upper = X_df.quantile(upper_q)

    return X_df.clip(lower=lower, upper=upper, axis=1)


def scale_features(X_df: pd.DataFrame) -> np.ndarray:
    """
    Standardize features.

    HDBSCAN is distance-based, so scaling is essential.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_scaled


# ============================================================
# Step 3: Run HDBSCAN
# ============================================================

def run_hdbscan(X_scaled: np.ndarray) -> HDBSCAN:
    """
    Run HDBSCAN clustering.

    Important parameters:

    min_cluster_size:
        Minimum number of rows needed to form a cluster.

    min_samples:
        Controls how conservative the model is.
        Higher values create more noise points.

    cluster_selection_method:
        'eom' means Excess of Mass, the standard HDBSCAN selection method.
    """
    model = HDBSCAN(
        min_cluster_size=300,
        min_samples=50,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
    )

    model.fit(X_scaled)

    return model


# ============================================================
# Step 4: Summarise results
# ============================================================

def create_hdbscan_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create cluster summary table.

    cluster = -1 means noise / outlier.
    """
    summary = (
        data.groupby("cluster")
        .agg(
            rows=("cluster", "size"),
            avg_cluster_probability=("cluster_probability", "mean"),
        )
        .reset_index()
    )

    summary["percent_of_data"] = 100 * summary["rows"] / len(data)
    summary["cluster_type"] = np.where(
        summary["cluster"] == -1,
        "noise_outlier",
        "cluster",
    )

    summary = summary[
        [
            "cluster",
            "cluster_type",
            "rows",
            "percent_of_data",
            "avg_cluster_probability",
        ]
    ]

    return summary


# ============================================================
# Step 5: Main
# ============================================================

def main() -> None:
    print("Starting HDBSCAN clustering pipeline...")
    print("Raw data folder:", RAW_DATA_DIR)

    df = load_all_csv_files(RAW_DATA_DIR)

    data, X_df = prepare_feature_matrix(df)

    X_clipped = robust_clip(X_df)
    X_scaled = scale_features(X_clipped)

    print("\nRunning HDBSCAN...")
    print("This may take a little time on 50,000+ rows.")

    model = run_hdbscan(X_scaled)

    data["cluster"] = model.labels_

    # In HDBSCAN:
    # cluster = -1 means noise / outlier.
    data["hdbscan_noise_outlier"] = data["cluster"] == -1

    # scikit-learn HDBSCAN provides probabilities_.
    # Higher value means stronger cluster membership.
    if hasattr(model, "probabilities_"):
        data["cluster_probability"] = model.probabilities_
    else:
        data["cluster_probability"] = np.nan

    summary = create_hdbscan_summary(data)

    clustered_output_path = PROCESSED_DATA_DIR / "clustered_time_series_jobs_hdbscan.csv"
    summary_output_path = OUTPUTS_DIR / "hdbscan_cluster_summary.csv"

    data.to_csv(clustered_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)

    print("\nDone.")
    print("Clustered HDBSCAN data saved to:", clustered_output_path)
    print("HDBSCAN cluster summary saved to:", summary_output_path)

    print("\nHDBSCAN cluster summary:")
    print(summary)

    n_clusters = data.loc[data["cluster"] != -1, "cluster"].nunique()
    n_noise = int((data["cluster"] == -1).sum())

    print("\nNumber of clusters excluding noise:", n_clusters)
    print("Number of noise/outlier rows:", n_noise)
    print("Noise/outlier percentage:", round(100 * n_noise / len(data), 2), "%")


if __name__ == "__main__":
    main()
