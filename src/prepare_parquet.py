from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DATA_DIR = ROOT_DIR / "data" / "demo" / "time_series_jobs"
PREPARED_DATA_DIR = ROOT_DIR / "data" / "prepared"
OUTPUT_PATH = PREPARED_DATA_DIR / "trip_features.parquet"


COLUMN_RENAME_MAP = {
    "md": "depth",
    "date_time_pd": "time",
    "ct_inc": "cont_vertical_orientation_incl",
    "ct_azi": "cont_horizontal_orientation",
    "st_inc": "static_vertical_orientation_incl",
    "st_azi": "static_horizontal_orientation",
    "cc_rpm": "pipe_rotation",
    "rpm_lt": "flow_rotation_lower",
    "rpm_ut": "flow_rotation_upper",
    "axlshkpeak": "axial_shock_peak",
    "axlshkrms": "axial_shock_rms",
    "radshkrms": "radial_shock_rms",
    "radshkpeak": "radial_shock_peak",
    "mwd_md": "ground_depth",
    "date_time_mwd": "ground_time",
    "mwd_incl": "ground_vertical_orientation",
    "mwd_azim": "ground_horizontal_orientation",
}


def extract_trip_id(file_name: str) -> int:
    """
    Extract the trip number from names such as time_series_job3.csv.
    """
    match = re.search(r"job(\d+)", file_name.lower())

    if match:
        return int(match.group(1))

    return -1


def load_raw_csv_files() -> pd.DataFrame:
    """
    Load and combine all four trip CSV files.
    """
    csv_files = sorted(SOURCE_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in:\n{SOURCE_DATA_DIR}"
        )

    frames = []

    for file_path in csv_files:
        print(f"Loading {file_path.name}...")

        trip_data = pd.read_csv(file_path)

        # Convert all source column names to lower case.
        trip_data.columns = [
            str(column).strip().lower()
            for column in trip_data.columns
        ]

        trip_data["source_file"] = file_path.name
        trip_data["trip_id"] = extract_trip_id(file_path.name)

        frames.append(trip_data)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nLoaded {len(csv_files)} trip files.")
    print(f"Combined shape: {combined.shape}")

    return combined


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the project-friendly feature names.
    """
    return data.rename(columns=COLUMN_RENAME_MAP)


def shift_datetimes_to_reference(
    data: pd.DataFrame,
    reference_start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Shift all datetime columns by one shared offset so the earliest
    available timestamp starts at 2020-01-01.

    Relative time differences are preserved.
    """
    datetime_columns = [
        column
        for column in ["time", "ground_time"]
        if column in data.columns
    ]

    for column in datetime_columns:
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        )

    valid_minimums = [
        data[column].min()
        for column in datetime_columns
        if data[column].notna().any()
    ]

    if not valid_minimums:
        print("No valid datetime values were found.")
        return data

    earliest_time = min(valid_minimums)
    reference_time = pd.Timestamp(reference_start)

    common_offset = reference_time - earliest_time

    for column in datetime_columns:
        data[column] = data[column] + common_offset

    print(f"Datetime offset applied: {common_offset}")

    return data


def add_raw_horizontal_orientation(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate raw horizontal orientation from triaxial gravity and
    magnetic-field measurements.

    No interpolation or missing-value filling is performed.
    """
    required_columns = {
        "gx",
        "gy",
        "gz",
        "bx",
        "by",
        "bz",
    }

    if not required_columns.issubset(data.columns):
        missing = sorted(required_columns.difference(data.columns))

        print(
            "Raw horizontal orientation was not calculated. "
            f"Missing columns: {missing}"
        )

        data["azim_raw"] = np.nan
        return data

    gx = pd.to_numeric(data["gx"], errors="coerce")
    gy = pd.to_numeric(data["gy"], errors="coerce")
    gz = pd.to_numeric(data["gz"], errors="coerce")

    bx = pd.to_numeric(data["bx"], errors="coerce")
    by = pd.to_numeric(data["by"], errors="coerce")
    bz = pd.to_numeric(data["bz"], errors="coerce")

    gravity_total = np.sqrt(
        gx**2
        + gy**2
        + gz**2
    )

    numerator = (
        gx * by
        - gy * bx
    ) * gravity_total

    denominator = (
        bz * (gx**2 + gy**2)
        - gz * (gx * bx + gy * by)
    )

    data["azim_raw"] = np.mod(
        np.degrees(
            np.arctan2(
                numerator,
                denominator,
            )
        ),
        360.0,
    )

    invalid_sensor_rows = (
        gx.isna()
        | gy.isna()
        | gz.isna()
        | bx.isna()
        | by.isna()
        | bz.isna()
    )

    data.loc[
        invalid_sensor_rows,
        "azim_raw",
    ] = np.nan

    return data


def circular_absolute_difference(
    angle_a: pd.Series,
    angle_b: pd.Series,
) -> pd.Series:
    """
    Calculate the smallest absolute difference between two angles.

    For example, 359 degrees and 1 degree differ by 2 degrees.
    """
    difference = (
        angle_a
        - angle_b
        + 180.0
    ) % 360.0 - 180.0

    return difference.abs()


def add_horizontal_orientation_difference(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the absolute circular difference between the ground
    horizontal orientation and raw horizontal orientation.
    """
    if (
        "ground_horizontal_orientation" not in data.columns
        or "azim_raw" not in data.columns
    ):
        data["horizontal_orientation_abs_difference"] = np.nan
        return data

    ground_orientation = pd.to_numeric(
        data["ground_horizontal_orientation"],
        errors="coerce",
    )

    raw_orientation = pd.to_numeric(
        data["azim_raw"],
        errors="coerce",
    )

    data["horizontal_orientation_abs_difference"] = (
        circular_absolute_difference(
            ground_orientation,
            raw_orientation,
        )
    )

    return data


def add_trip_sequence_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort each trip and create a row sequence number.
    """
    sort_columns = ["trip_id"]

    if "time" in data.columns and data["time"].notna().any():
        sort_columns.append("time")
    elif "depth" in data.columns:
        sort_columns.append("depth")

    data = data.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    data["row_in_trip"] = (
        data.groupby("trip_id")
        .cumcount()
    )

    return data


def convert_numeric_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert expected sensor and operational columns to numeric values.

    Missing measurements remain missing.
    """
    non_numeric_columns = {
        "source_file",
        "time",
        "ground_time",
    }

    for column in data.columns:
        if column not in non_numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def print_dataset_summary(
    data: pd.DataFrame,
) -> None:
    """
    Print a short summary of the prepared dataset.
    """
    print("\nPrepared dataset summary")
    print("------------------------")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Trips: {data['trip_id'].nunique()}")

    print("\nRows per trip:")
    print(
        data["trip_id"]
        .value_counts()
        .sort_index()
    )

    print("\nFeature sampling coverage:")

    coverage = (
        100.0
        * data.notna().mean()
    ).sort_values(ascending=False)

    print(coverage.round(2).to_string())


def main() -> None:
    PREPARED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_raw_csv_files()

    data = rename_columns(data)

    data = shift_datetimes_to_reference(data)

    data = convert_numeric_columns(data)

    data = add_raw_horizontal_orientation(data)

    data = add_horizontal_orientation_difference(data)

    data = add_trip_sequence_columns(data)

    print_dataset_summary(data)

    data.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    print(f"\nPrepared Parquet file saved to:\n{OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()