from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT_DIR = Path(__file__).resolve().parents[1]

CLUSTERED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs_hdbscan.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "hdbscan_job_cluster_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def prepare_data() -> pd.DataFrame:
    """
    Load HDBSCAN clustered data and prepare job number and x-axis.
    """
    if not CLUSTERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{CLUSTERED_DATA_PATH}\n\n"
            "Run this first:\npython src\\cluster_time_series_hdbscan.py"
        )

    df = pd.read_csv(CLUSTERED_DATA_PATH)

    required_columns = FEATURES_USED_FOR_CLUSTERING + ["cluster"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "job_number" not in df.columns:
        if "source_file" not in df.columns:
            raise ValueError("Missing both 'job_number' and 'source_file'.")

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
    Choose x-axis in this priority:

    1. date_time_pd
    2. MD
    3. row number within job
    """
    if "date_time_pd" in job_df.columns and job_df["date_time_pd"].notna().sum() > 0:
        return job_df["date_time_pd"], "date_time_pd"

    if "MD" in job_df.columns:
        return job_df["MD"], "MD"

    return job_df["row_in_job"], "row_in_job"


def create_cluster_color_mapping(cluster_values: list[int]) -> dict[int, tuple]:
    """
    Create color mapping for HDBSCAN clusters.

    cluster = -1 is HDBSCAN noise/outlier.
    """
    cmap = plt.colormaps.get_cmap("tab20")

    normal_clusters = [cluster for cluster in cluster_values if cluster != -1]

    cluster_to_color = {}

    # Noise/outlier as black/dark marker
    if -1 in cluster_values:
        cluster_to_color[-1] = "black"

    for i, cluster in enumerate(normal_clusters):
        cluster_to_color[cluster] = cmap(i % 20)

    return cluster_to_color


def plot_one_job(job_df: pd.DataFrame, job_number: int, cluster_values: list[int]) -> None:
    """
    Create stacked time-series scatter plots for one job.
    Each feature is shown in a separate subplot.
    Points are colored by HDBSCAN cluster.
    """
    x, x_label = choose_x_axis(job_df)

    cluster_to_color = create_cluster_color_mapping(cluster_values)

    fig, axes = plt.subplots(
        nrows=len(FEATURES_USED_FOR_CLUSTERING),
        ncols=1,
        figsize=(18, 2.4 * len(FEATURES_USED_FOR_CLUSTERING)),
        sharex=True,
    )

    for ax, feature in zip(axes, FEATURES_USED_FOR_CLUSTERING):
        for cluster in cluster_values:
            mask = job_df["cluster"] == cluster

            label = "Noise / Outlier" if cluster == -1 else f"Cluster {cluster}"

            ax.scatter(
                x[mask],
                job_df.loc[mask, feature],
                s=8,
                alpha=0.75,
                color=cluster_to_color[cluster],
                label=label,
            )

        ax.set_ylabel(feature, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)

    legend_elements = []

    for cluster in cluster_values:
        label = "Noise / Outlier" if cluster == -1 else f"Cluster {cluster}"

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=label,
                markerfacecolor=cluster_to_color[cluster],
                markeredgecolor=cluster_to_color[cluster],
                markersize=7,
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=min(len(cluster_values), 6),
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8,
    )

    fig.suptitle(
        f"Job {job_number} - HDBSCAN Sensor Time-Series Colored by Cluster",
        fontsize=14,
        y=0.999,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])

    output_file = OUTPUT_DIR / f"job_{job_number}_hdbscan_clustered_time_series.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def main() -> None:
    print("Loading HDBSCAN clustered data...")

    df = prepare_data()

    cluster_values = sorted(df["cluster"].dropna().unique())
    job_numbers = sorted(df["job_number"].dropna().unique())

    print("Detected clusters:", cluster_values)
    print("Detected job numbers:", job_numbers)

    for job_number in job_numbers:
        job_df = df[df["job_number"] == job_number].copy()

        if job_df.empty:
            continue

        print(f"Plotting job {job_number} with {len(job_df)} rows...")

        plot_one_job(
            job_df=job_df,
            job_number=int(job_number),
            cluster_values=cluster_values,
        )

    print("\nDone.")
    print("HDBSCAN plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
