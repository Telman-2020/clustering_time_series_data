# Multivariate Time-Series Behaviour Segmentation

**Telman Maghrebi** |
Senior Data Scientist |
Contact: [telman_mgh@yahoo.com](mailto:telman_mgh@yahoo.com)

## Overview

This project is a proof of concept for identifying behavioural and operational regimes in multivariate time-series sensor data using two unsupervised clustering methods:

1. **KMeans**
2. **HDBSCAN**

The clustering pipeline consumes a prepared Parquet feature dataset containing multiple time-sequenced trip recordings.

The main objective is to separate recurring operating behaviours using acceleration, magnetic-field, rotational, pipe-rotation, flow-induced rotation, orientation, shock, vibration, depth, and related operational measurements.

The resulting regime labels can support:

* Behaviour segmentation
* Operating-state identification
* Noise and anomaly isolation
* Regime-specific denoising
* Control-engineering decisions
* Trip-level behavioural comparison
* Downstream machine-learning pipelines

## Problem Statement

Industrial time-series datasets commonly contain several operating behaviours within the same continuous recording.

A single trip may include:

* Stable motion
* Changing motion
* Low- and high-vibration periods
* Rotational transitions
* Flow-driven changes
* Sensor disturbances
* Short abnormal intervals
* Repeated operating patterns

Processing the entire recording as one homogeneous signal can mix distinct operating conditions together. This can reduce the effectiveness of filtering, denoising, monitoring, control logic, and downstream models.

This project segments each trip into more consistent behavioural regimes using relationships between multiple sensor and operational channels.

## Project Architecture

The main project workflow is:

```text
Prepared Parquet feature dataset
                ↓
Dataset validation
                ↓
Feature selection
                ↓
Finite-value row selection
                ↓
Robust quantile clipping
                ↓
Standard scaling
                ↓
KMeans clustering
                ↓
HDBSCAN clustering
                ↓
Behavioural-regime comparison
                ↓
Anomaly/noise inspection
                ↓
Static and interactive visualisation
```

The clustering and dashboard applications do not read raw CSV files directly. They use:

```text
data/prepared/trip_features.parquet
```

Raw datasets, prepared Parquet files, processed outputs, figures, and other generated files are excluded from Git through `.gitignore`.

## Prepared Dataset

The prepared Parquet dataset contains:

* Four trip recordings
* Time and depth sequence information
* Sensor and operational features
* A trip identifier
* A sequence number within each trip
* Derived horizontal-orientation measurements
* Derived circular orientation differences

The expected prepared-data path is:

```text
data/prepared/trip_features.parquet
```

The prepared dataset is validated by:

```text
src/prepared_data.py
```

This module checks that the required fields exist and provides the default clustering feature list.

## Sensor and Operational Features

### Triaxial Acceleration

The acceleration channels are:

```text
gx
gy
gz
```

They help describe:

* Motion intensity
* Directional changes
* Vibration behaviour
* Impacts
* Changes in lateral and vertical movement

### Triaxial Magnetic Field

The magnetic-field channels are:

```text
bx
by
bz
```

They help describe:

* Directional behaviour
* Horizontal-orientation changes
* Magnetic disturbances
* Rotational state
* Positional alignment changes

### Sesor Rotational Chassis Channels

The rotational channels include:

```text
rx
ry
```

They help identify:

* Rotational transitions
* Stable and unstable rotational behaviour
* Directional movement changes
* Differences between operating states

### Pipe Rotation

The primary pipe-rotation feature is:

```text
pipe_rotation
```

It helps identify:

* Stationary periods
* Low-rotation regimes
* High-rotation regimes
* Rotational transitions
* Operating-state changes

### Flow-Induced Rotation

The prepared dataset contains:

```text
flow_rotation_lower
flow_rotation_upper
```

These measurements can help distinguish:

* Flow-driven operating regimes
* Mechanical rotation from flow-related rotation
* Stable flow behaviour
* Irregular flow-induced movement

### Shock and Vibration

The available shock and vibration features include:

```text
axial_shock_peak
axial_shock_rms
radial_shock_peak
radial_shock_rms
```

These features help identify:

* High-impact events
* Mechanical instability
* Sustained vibration
* Sudden operating changes
* Noisy or abnormal regimes

Some features may have lower sampling coverage than the main sensor channels. The pipeline does not interpolate or fill missing measurements. A row is eligible for clustering only when all currently selected model features contain valid finite values.

### Depth and Orientation

Depth and orientation-related fields include:

```text
depth
ground_depth
cont_vertical_orientation_incl
cont_horizontal_orientation
static_vertical_orientation_incl
static_horizontal_orientation
ground_vertical_orientation
ground_horizontal_orientation
azim_raw
horizontal_orientation_abs_difference
```

These measurements provide spatial and directional context for each trip.

## Default Clustering Features

By default, both clustering algorithms use the same features:

```text
gx
gy
gz
bx
by
bz
rx
ry
pipe_rotation
flow_rotation_lower
flow_rotation_upper
```

Using the same eligible rows and scaled feature matrix enables a fairer comparison between KMeans and HDBSCAN.

Additional numeric features can be selected from the dashboard.

## Feature Preparation

The shared model preparation performs the following operations:

1. Confirms that the selected features exist.
2. Converts selected features to numeric values.
3. Replaces infinite values with missing values.
4. Excludes rows that are missing any selected feature.
5. Clips extreme values using feature-level quantiles.
6. Standardises the selected features using `StandardScaler`.

The pipeline intentionally performs no:

* Linear interpolation
* Forward filling
* Backward filling
* Median imputation
* Synthetic replacement of missing sensor measurements

## KMeans

KMeans is a centroid-based clustering algorithm.

It divides the feature space into a predefined number of clusters. Each observation is assigned to its nearest cluster centre.

### Main Parameters

* `n_clusters`: number of clusters
* `max_iter`: maximum number of optimisation iterations
* `tolerance`: convergence tolerance
* `random_state`: reproducible initialisation seed
* `outlier_quantile`: cluster-distance threshold for possible outliers

### Strengths

* Fast and widely understood
* Easy to reproduce
* Suitable as a baseline model
* Produces a fixed number of groups
* Works well for compact, similarly scaled clusters

### Limitations

* The cluster count must be selected in advance
* Every observation is assigned to a cluster
* It does not naturally create a separate noise class
* It can be sensitive to outliers
* Irregularly shaped regimes may be represented poorly

### KMeans Outlier Flag

KMeans does not natively detect noise.

This project calculates the distance between each observation and its assigned centroid. Observations above the selected distance quantile are flagged as:

```text
possible_outlier
```

This flag is a distance-based diagnostic and is not a separate KMeans cluster.

## HDBSCAN

HDBSCAN is a density-based clustering algorithm.

It identifies stable dense regions in the feature space and can leave sparse or unstable observations unassigned.

Native HDBSCAN noise observations use the label:

```text
-1
```

The dashboard displays this label as:

```text
Anomaly / Noise
```

### Main Parameters

* `min_cluster_size`: minimum size of a stable cluster
* `min_samples`: controls the conservativeness of noise detection
* `metric`: distance metric
* `cluster_selection_method`: broader EOM or finer leaf selection
* `cluster_selection_epsilon`: optional nearby-cluster merging threshold
* `target_total_labels`: dashboard comparison-oriented displayed label count

### Strengths

* Does not require a fixed native cluster count
* Supports irregular cluster shapes
* Naturally identifies noise
* Can represent changing local density
* Useful for unstable or sparse operating behaviour

### Limitations

* Sensitive to parameter selection
* Can classify a large proportion of observations as noise
* Density separation becomes more difficult in high dimensions
* Cluster meaning requires engineering interpretation
* Displayed target labels involve post-processing and are not native HDBSCAN outputs

## HDBSCAN Displayed Regimes

The project preserves the original HDBSCAN labels in:

```text
hdbscan_original_cluster
```

For dashboard comparison, native HDBSCAN clusters may be merged or split to reach a selected displayed regime count.

The native noise label `-1` is always preserved and is never converted into a normal behavioural regime.

## Project Structure

```text
clustering_time_series_data/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── prepared/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── prepare_parquet.py
│   ├── prepared_data.py
│   ├── cluster_time_series_methods.py
│   ├── plot_kmeans_hdbscan_time_series.py
│   └── dash_cluster_app.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Main Python Modules

### `prepare_parquet.py`

Creates the prepared Parquet dataset from locally available trip CSV files.

It:

* Combines the trip recordings
* Standardises column names
* Renames source fields
* Converts expected numeric features
* Shifts timestamps to a common reference date
* Calculates raw horizontal orientation
* Calculates circular orientation difference
* Adds trip and within-trip sequence identifiers
* Writes the prepared Parquet file

Raw input files are not included in the repository.

### `prepared_data.py`

Provides reusable prepared-data loading and validation.

It:

* Loads the Parquet dataset
* Validates required fields
* Parses datetime fields
* Sorts rows by trip and sequence
* Returns available numeric clustering features
* Defines the default clustering inputs

### `cluster_time_series_methods.py`

Runs the complete clustering workflow.

It:

* Loads the prepared Parquet dataset
* Creates the shared feature matrix
* Runs KMeans
* Runs HDBSCAN
* Creates cluster summaries
* Saves processed Parquet outputs
* Saves CSV comparison reports

### `plot_kmeans_hdbscan_time_series.py`

Creates static Matplotlib figures for both methods.

It supports:

* One selected trip or all trips
* Configurable x-axis
* Configurable plotted features
* Point downsampling
* PNG output

### `dash_cluster_app.py`

Provides the interactive Dash application.

It supports:

* Side-by-side KMeans and HDBSCAN plots
* Trip filtering
* X-axis selection
* Model feature selection
* Plot feature selection
* KMeans parameter controls
* HDBSCAN parameter controls
* Cluster filtering
* Cluster summary tables
* Anomaly/noise highlighting
* Interactive model reruns
* Shared feature preparation for both algorithms

## Installation

Create and activate a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The main dependencies are:

* NumPy
* pandas
* Matplotlib
* scikit-learn
* Plotly
* Dash
* PyArrow

## Preparing the Parquet Dataset

Place the local trip CSV files in:

```text
data/raw/time_series_jobs/
```

Then run:

```powershell
python src\prepare_parquet.py
```

The script creates:

```text
data/prepared/trip_features.parquet
```

The raw and prepared datasets are ignored by Git.

When a prepared Parquet file is already available, place it directly at the expected path and skip the CSV preparation command.

## Validate the Prepared Dataset

Run:

```powershell
python src\prepared_data.py
```

This prints:

* Dataset shape
* Trip identifiers
* Default clustering features
* Available numeric features
* Sampling coverage for each feature

## Run the Clustering Pipeline

Run:

```powershell
python src\cluster_time_series_methods.py
```

The pipeline creates:

```text
data/processed/kmeans_clustered_trip_features.parquet
data/processed/hdbscan_clustered_trip_features.parquet
outputs/kmeans_cluster_summary.csv
outputs/hdbscan_cluster_summary.csv
outputs/clustering_method_comparison.csv
```

Generated outputs are ignored by Git.

## Generate Static Figures

Generate figures for all trips:

```powershell
python src\plot_kmeans_hdbscan_time_series.py
```

Generate figures for one trip:

```powershell
python src\plot_kmeans_hdbscan_time_series.py --trip 1
```

Select a different x-axis:

```powershell
python src\plot_kmeans_hdbscan_time_series.py --trip 1 --x-axis time
```

Select specific features:

```powershell
python src\plot_kmeans_hdbscan_time_series.py `
  --trip 1 `
  --features gx gy gz pipe_rotation
```

Figures are saved under:

```text
outputs/figures/
```

## Launch the Dashboard

Run:

```powershell
python src\dash_cluster_app.py
```

Open:

```text
http://127.0.0.1:8050/
```

The dashboard initially runs both algorithms using the default features and parameters.

The models can then be rerun interactively using different:

* Model features
* Cluster counts
* KMeans convergence settings
* KMeans outlier thresholds
* HDBSCAN density settings
* HDBSCAN distance metrics
* HDBSCAN cluster-selection settings
* Displayed regime counts

## Interpretation

KMeans provides a fixed partition of the eligible observations. Every observation receives a normal cluster label.

HDBSCAN identifies stable density-based regimes and preserves a separate anomaly/noise class for observations outside those regimes.

The clusters are behavioural groupings rather than automatically named engineering states. Their physical meaning should be interpreted using:

* Feature distributions
* Sensor trends
* Trip location
* Time and depth sequence
* Operating context
* Domain knowledge

A cluster label should not be treated as a confirmed physical operating state until it has been reviewed and validated.

## Real-Time Application

The intended real-time extension is to use the inferred behavioural regime as an additional processing or control signal.

A possible online workflow is:

1. Receive the latest sensor observation or time window.
2. Apply the same feature definitions used during training.
3. Apply the saved clipping and scaling parameters.
4. Assign the observation to an operating regime.
5. Apply regime-specific filtering, thresholds, diagnostics, or control logic.

Potential applications include:

* Smoother filtering in stable regimes
* Stronger noise suppression in high-vibration regimes
* Transition-specific control logic
* Separate handling of anomaly/noise observations
* Regime-specific thresholds
* Behaviour-aware downstream models

The current repository is a proof of concept and does not yet implement a complete production online-inference service.

## Current Example Result

Using the default feature configuration on the prepared four-trip dataset:

```text
Rows modelled: 50,083
KMeans displayed clusters: 7
HDBSCAN normal displayed regimes: 6
HDBSCAN anomaly/noise rows: 14,539
HDBSCAN anomaly/noise percentage: 29.03%
```

These values are dataset- and parameter-dependent and may change when the selected features or hyperparameters change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
