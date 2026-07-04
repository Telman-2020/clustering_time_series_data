from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

PREPARED_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "prepared"
    / "trip_features.parquet"
)


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


REQUIRED_COLUMNS = {
    "trip_id",
    "row_in_trip",
    "time",
    "depth",
    *DEFAULT_CLUSTERING_FEATURES,
}


NON_MODEL_COLUMNS = {
    "source_file",
    "trip_id",
    "row_in_trip",
    "time",
    "ground_time",
}


def load_prepared_data() -> pd.DataFrame:
    """
    Load and validate the prepared Parquet feature dataset.
    """
    if not PREPARED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared Parquet file was not found:\n"
            f"{PREPARED_DATA_PATH}\n\n"
            "Run this command first:\n"
            "python src\\prepare_parquet.py"
        )

    data = pd.read_parquet(
        PREPARED_DATA_PATH,
        engine="pyarrow",
    )

    if data.empty:
        raise ValueError("The prepared Parquet dataset is empty.")

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "The prepared dataset is missing required columns: "
            f"{missing_columns}"
        )

    data["time"] = pd.to_datetime(
        data["time"],
        errors="coerce",
    )

    if "ground_time" in data.columns:
        data["ground_time"] = pd.to_datetime(
            data["ground_time"],
            errors="coerce",
        )

    data = data.sort_values(
        ["trip_id", "row_in_trip"],
        kind="stable",
    ).reset_index(drop=True)

    return data


def get_available_numeric_features(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return prepared numeric columns that can be selected for clustering.
    """
    return [
        column
        for column in data.columns
        if column not in NON_MODEL_COLUMNS
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def validate_default_features(
    data: pd.DataFrame,
) -> None:
    """
    Confirm that all default clustering features are available.
    """
    missing_features = [
        feature
        for feature in DEFAULT_CLUSTERING_FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Default clustering features are missing: "
            f"{missing_features}"
        )


def main() -> None:
    data = load_prepared_data()
    validate_default_features(data)

    available_features = get_available_numeric_features(data)

    print("Prepared dataset loaded successfully.")
    print("Shape:", data.shape)
    print("Trips:", sorted(data["trip_id"].unique()))
    print("Default clustering features:")
    print(DEFAULT_CLUSTERING_FEATURES)

    print("\nAll selectable numeric features:")
    for feature in available_features:
        coverage = 100 * data[feature].notna().mean()
        print(f" - {feature}: {coverage:.2f}% coverage")


if __name__ == "__main__":
    main()