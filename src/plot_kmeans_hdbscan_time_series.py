from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASETS = {
    "KMeans": {
        "path": ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs_kmeans.csv",
        "fallback_path": ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs.csv",
        "output_dir": ROOT_DIR / "outputs" / "kmeans_time_series_plots",
        "title_prefix": "KMeans",
        "noise_cluster": None,
    },
    "HDBSCAN merged to 7 clusters": {
        "path": ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs_hdbscan_merged.csv",
        "fallback_path": ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs_hdbscan.csv",
        "output_dir": ROOT_DIR / "outputs" / "hdbscan_merged_time_series_plots",
        "title_prefix": "HDBSCAN merged to 7 clusters",
        "noise_cluster": -1,
    },
}

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
# Helpers
# ============================================================

def extract_job_number_from_source_file(source_file: str) -> int:
    """
    Extract job number from names such as:

    time_series_job1.csv
    time_series_job2.csv
    """
    match = re.search(r"job(\d+)", str(source_file).lower())

    if match:
        return int(match.group(1))

    return -1


def load_clustered_data(path: Path, fallback_path: Path, method_name: str) -> pd.DataFrame:
    """
    Load clustered data for one method.
    """
    selected_path = path

    if not selected_path.exists() and fallback_path.exists():
        selected_path = fallback_path

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Could not find clustered data for {method_name}.\n"
            f"Expected:\n{path}\n\n"
            "Run this first:\npython src/cluster_time_series_methods.py"
        )

    print(f"Loading {method_name}: {selected_path}")
    df = pd.read_csv(selected_path)

    required_columns = FEATURES_USED_FOR_CLUSTERING + ["cluster"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{method_name} data is missing required columns: {missing_columns}")

    if "job_number" not in df.columns:
        if "source_file" not in df.columns:
            raise ValueError(
                f"{method_name} data does not contain 'job_number' or 'source_file'."
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

    return df


def choose_x_axis(job_df: pd.DataFrame):
    """
    Choose best x-axis:
    1. date_time_pd
    2. MD
    3. row number inside job
    """
    if "date_time_pd" in job_df.columns and job_df["date_time_pd"].notna().sum() > 0:
        return job_df["date_time_pd"], "date_time_pd"

    if "MD" in job_df.columns:
        return job_df["MD"], "MD"

    return job_df["row_in_job"], "row_in_job"


def create_cluster_color_mapping(cluster_values: list[int], noise_cluster: int | None = None) -> dict[int, object]:
    """
    Create color mapping for clusters.

    For HDBSCAN, noise cluster -1 is shown in black.
    """
    cmap = plt.colormaps.get_cmap("tab20")

    cluster_to_color = {}

    normal_clusters = cluster_values

    if noise_cluster is not None and noise_cluster in cluster_values:
        cluster_to_color[noise_cluster] = "black"
        normal_clusters = [cluster for cluster in cluster_values if cluster != noise_cluster]

    for i, cluster in enumerate(normal_clusters):
        cluster_to_color[cluster] = cmap(i % 20)

    return cluster_to_color


def cluster_label(cluster: int, noise_cluster: int | None = None) -> str:
    """
    Human-readable cluster label.
    """
    if noise_cluster is not None and cluster == noise_cluster:
        return "Noise / Outlier (-1)"

    return f"Cluster {cluster}"


def plot_one_job(
    job_df: pd.DataFrame,
    job_number: int,
    cluster_values: list[int],
    method_name: str,
    output_dir: Path,
    noise_cluster: int | None = None,
) -> None:
    """
    Create one stacked plot figure for one job.
    Each subplot shows one feature.
    Points are coloured by cluster.
    """
    x, x_label = choose_x_axis(job_df)
    cluster_to_color = create_cluster_color_mapping(cluster_values, noise_cluster=noise_cluster)

    fig, axes = plt.subplots(
        nrows=len(FEATURES_USED_FOR_CLUSTERING),
        ncols=1,
        figsize=(18, 2.4 * len(FEATURES_USED_FOR_CLUSTERING)),
        sharex=True,
    )

    for ax, feature in zip(axes, FEATURES_USED_FOR_CLUSTERING):
        for cluster in cluster_values:
            mask = job_df["cluster"] == cluster

            ax.scatter(
                x[mask],
                job_df.loc[mask, feature],
                s=8,
                alpha=0.75,
                color=cluster_to_color[cluster],
            )

        ax.set_ylabel(feature, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=cluster_label(cluster, noise_cluster=noise_cluster),
            markerfacecolor=cluster_to_color[cluster],
            markeredgecolor=cluster_to_color[cluster],
            markersize=7,
        )
        for cluster in cluster_values
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=min(len(cluster_values), 7),
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8,
    )

    fig.suptitle(
        f"Job {job_number} - {method_name} Sensor Time-Series Colored by Cluster",
        fontsize=14,
        y=0.999,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])

    safe_method_name = (
        method_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    output_file = output_dir / f"job_{job_number}_{safe_method_name}_time_series.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def plot_method(method_name: str, config: dict) -> None:
    """
    Plot all jobs for one clustering method.
    """
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_clustered_data(
        path=config["path"],
        fallback_path=config["fallback_path"],
        method_name=method_name,
    )

    cluster_values = sorted(int(c) for c in df["cluster"].dropna().unique())
    job_numbers = sorted(int(j) for j in df["job_number"].dropna().unique())

    print(f"\n{method_name}")
    print("Detected clusters:", cluster_values)
    print("Detected job numbers:", job_numbers)

    for job_number in job_numbers:
        job_df = df[df["job_number"] == job_number].copy()

        if job_df.empty:
            continue

        print(f"Plotting {method_name}, job {job_number}, rows={len(job_df)}")

        plot_one_job(
            job_df=job_df,
            job_number=job_number,
            cluster_values=cluster_values,
            method_name=config["title_prefix"],
            output_dir=output_dir,
            noise_cluster=config["noise_cluster"],
        )

    print(f"{method_name} plots saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Creating KMeans and HDBSCAN time-series plots...")

    for method_name, config in DATASETS.items():
        plot_method(method_name, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
