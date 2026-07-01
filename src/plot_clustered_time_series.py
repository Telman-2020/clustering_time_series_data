from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT_DIR = Path(__file__).resolve().parents[1]

CLUSTERED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clustered_time_series_jobs.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "job_cluster_plots"

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
    Extract job number from names like:

    time_series_job1.csv
    time_series_job2.csv
    """
    match = re.search(r"job(\d+)", str(source_file).lower())

    if match:
        return int(match.group(1))

    return -1


def prepare_data() -> pd.DataFrame:
    """
    Load clustered data and prepare job number and x-axis columns.
    """
    if not CLUSTERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find clustered data file:\n{CLUSTERED_DATA_PATH}\n\n"
            "Run this first:\npython src\\cluster_time_series.py"
        )

    df = pd.read_csv(CLUSTERED_DATA_PATH)

    required_columns = FEATURES_USED_FOR_CLUSTERING + ["cluster"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "job_number" not in df.columns:
        if "source_file" not in df.columns:
            raise ValueError(
                "The data does not contain 'job_number' or 'source_file'. "
                "Cannot identify jobs."
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


def plot_one_job(job_df: pd.DataFrame, job_number: int, cluster_values: list[int]) -> None:
    """
    Create one stacked plot figure for one job.
    Each subplot shows one feature.
    Points are colored by cluster.
    """
    x, x_label = choose_x_axis(job_df)

    cmap = plt.colormaps.get_cmap("tab10")
    cluster_to_color = {
        cluster: cmap(i % 10)
        for i, cluster in enumerate(cluster_values)
    }

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
            label=f"Cluster {cluster}",
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
    )

    fig.suptitle(
        f"Job {job_number} - Sensor Time-Series Colored by Cluster",
        fontsize=14,
        y=0.999,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = OUTPUT_DIR / f"job_{job_number}_clustered_time_series.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def main() -> None:
    print("Loading clustered data...")
    df = prepare_data()

    cluster_values = sorted(df["cluster"].dropna().unique())

    print("Detected clusters:", cluster_values)
    print("Detected job numbers:", sorted(df["job_number"].dropna().unique()))

    for job_number in sorted(df["job_number"].dropna().unique()):
        job_df = df[df["job_number"] == job_number].copy()

        if job_df.empty:
            continue

        plot_one_job(
            job_df=job_df,
            job_number=int(job_number),
            cluster_values=cluster_values,
        )

    print("\nDone.")
    print("Plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
