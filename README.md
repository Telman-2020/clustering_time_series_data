Telman Maghrebi  
Senior Data Scientist  
Email to contact: telman_mgh@yahoo.com  

# Multivariate Time-Series Behaviour Segmentation

This project is a proof of concept for segmenting behavioural patterns in multivariate time-series sensor data using **KMeans** and **HDBSCAN**.

The dataset contains four trip recordings. Each trip includes time-sequenced measurements from:

- A triaxial accelerometer
- A triaxial magnetometer
- A triaxial gyroscope
- Pipe rotation measurements
- Flow-induced rotational measurements
- Shock and vibration measurements
- Depth and orientation-related measurements
- Other operational sensor channels

The main objective is to identify and separate recurring behavioural regimes within each trip using positional and rotational sensor data.

These segmented regimes can then be used in real time to:

- Separate high-frequency operating behaviours
- Identify stable and unstable regimes
- Detect noisy or abnormal sensor patterns
- Support denoising strategies
- Improve control engineering decisions
- Create regime-specific processing rules
- Reduce the effect of mixed operating conditions on downstream algorithms

## Problem Statement

Industrial time-series data often contains several operating behaviours within the same trip.

A single trip may include:

- Stable motion
- Changing motion
- High vibration
- Low vibration
- Rotational transitions
- Flow-driven changes
- Sensor noise
- Short abnormal intervals
- Repeated operating patterns

When all observations are processed as one continuous signal, different behaviours become mixed together. This can make denoising, monitoring, and control logic less effective.

The purpose of this project is to segment each trip into behavioural regimes based on the relationships between acceleration, magnetic field, rotational movement, pipe rotation, flow-induced rotation, and other sensor measurements.

The resulting regime labels can be used to separate high-frequency data into more consistent groups before applying filtering, denoising, control logic, or further machine-learning models.

## Project Objective

The project compares two unsupervised clustering methods:

1. **KMeans**
2. **HDBSCAN**

Both methods are applied to the same selected and scaled feature matrix so that their outputs can be compared fairly.

The project aims to determine which method is more suitable for:

- Behaviour segmentation
- Operational regime identification
- Noise isolation
- Trip-level pattern comparison
- Real-time high-frequency data separation

## KMeans

KMeans is a centroid-based clustering algorithm.

It divides the data into a predefined number of groups. Each observation is assigned to the nearest cluster centre.

### Important Parameters

- `k`: number of clusters
- `max_iter`: maximum number of update iterations
- `tol`: convergence tolerance
- `random_state`: controls reproducible initialisation
- `outlier_quantile`: distance threshold used to flag observations far from their assigned centroid

### Strengths

- Simple and fast
- Easy to interpret
- Suitable for compact and well-separated regimes
- Useful as a baseline method
- Produces a fixed number of clusters

### Limitations

- The number of clusters must be chosen in advance
- Every observation must belong to a cluster
- It does not naturally create a separate noise class
- It can be sensitive to outliers
- It works best when clusters are approximately compact and similar in scale
- Complex or irregular behavioural regimes may be forced into unsuitable groups

## HDBSCAN

HDBSCAN is a density-based clustering algorithm.

It identifies stable dense regions in the feature space and separates observations that do not belong to a sufficiently stable region.

These unassigned observations are labelled as:

```text
Anomaly / Noise
```

The anomaly/noise group is always shown in black in the dashboard.

### Important Parameters

- `min_cluster_size`: smallest group allowed to form a cluster
- `min_samples`: controls how conservative the noise detection is
- `metric`: distance metric used between observations
- `cluster_selection_method`: controls broader or finer cluster selection
- `cluster_selection_epsilon`: optional threshold for merging nearby clusters
- `target total displayed labels`: comparison-oriented post-processing used by the dashboard

### Strengths

- Does not require a fixed number of native clusters
- Can identify irregular behavioural regimes
- Can isolate sparse or unstable observations
- Provides a natural anomaly/noise class
- Better suited to operating data with changing local density

### Limitations

- Sensitive to parameter selection
- May classify too many observations as noise
- High-dimensional feature spaces can reduce density separation
- Results still require engineering interpretation
- The displayed target cluster count is post-processing and not native HDBSCAN behaviour

## Sensor Data

The four trips contain positional, rotational, and operational sensor measurements recorded as time-series sequences.

The main sensor groups are:

### Triaxial Accelerometer

The accelerometer measures acceleration along three orthogonal axes.

These channels help describe:

- Motion intensity
- Directional changes
- Vibration behaviour
- Impact-related patterns
- Changes in vertical and lateral movement

### Triaxial Magnetometer

The magnetometer measures the magnetic field along three orthogonal axes.

These channels help describe:

- Directional behaviour
- Horizontal orientation changes
- Magnetic disturbances
- Rotational state
- Changes in positional alignment

### Triaxial Gyroscope

The gyroscope measures angular movement around three axes.

These measurements help identify:

- Rotational transitions
- Angular velocity changes
- Stable and unstable rotational behaviour
- Changes in directional movement

### Pipe Rotation

Pipe rotation represents the rotational speed or activity of the main rotating structure.

It is important for identifying:

- Stationary periods
- Low-rotation regimes
- High-rotation regimes
- Rotational transitions
- Changes in operating state

### Flow-Induced Rotation

Flow-induced rotational measurements represent rotational behaviour associated with fluid flow or internal rotating components.

These measurements can help separate:

- Flow-driven operating regimes
- Mechanical rotation from flow-related rotation
- Stable flow behaviour
- Irregular flow-induced motion

### Shock and Vibration

Shock and vibration measurements describe short-duration impacts and sustained vibration energy.

They are useful for detecting:

- High-impact events
- Mechanical instability
- Sudden changes in operating behaviour
- Noisy or abnormal regimes

### Depth and Orientation Measurements

Depth and orientation-related measurements provide the spatial and directional context of each trip.

They help relate behavioural changes to:

- Position along the trip
- Directional changes
- Vertical orientation
- Horizontal orientation
- Static and continuous measurement differences

## Behaviour Segmentation Workflow

The workflow is:

```text
Four trip datasets
        ↓
Time-series alignment by recorded sequence
        ↓
Feature selection
        ↓
Numeric conversion and validity checks
        ↓
Feature scaling
        ↓
KMeans clustering
        ↓
HDBSCAN clustering
        ↓
Behavioural regime comparison
        ↓
Anomaly/noise inspection
        ↓
Trip-level visualisation
        ↓
Real-time regime-based processing
```

## Default Clustering Inputs

By default, the clustering models use:

- Triaxial accelerometer data
- Triaxial magnetometer data
- Gyroscope-related rotational channels
- Pipe rotation
- Lower flow-induced rotation
- Upper flow-induced rotation

These channels provide the main description of positional, magnetic, and rotational behaviour.

Additional operational features can be selected from the dashboard when required.

## Dashboard

The Dash application provides an interactive comparison of KMeans and HDBSCAN.

It includes:

- Side-by-side KMeans and HDBSCAN charts
- Trip selection
- X-axis selection
- Model input feature selection
- Plot feature selection
- KMeans hyperparameter controls
- HDBSCAN hyperparameter controls
- Cluster filters
- Cluster summary tables
- Anomaly/noise highlighting
- Automatic rerunning of both algorithms using the same selected inputs
- Per-trip behavioural regime inspection

The dashboard is designed to help compare:

- How each method segments the same trip
- Whether regimes are stable across different sensors
- Which observations are identified as anomaly/noise
- How regime boundaries change with hyperparameters
- Whether HDBSCAN isolates noise more effectively than KMeans

## Real-Time Application

The intended real-time application is to use the cluster or regime label as an additional control signal.

For each incoming high-frequency observation:

1. The current sensor values are processed.
2. The behavioural regime is identified.
3. The high-frequency data is assigned to the corresponding regime.
4. A regime-specific denoising or control strategy can be applied.

This allows the control system to treat different operating conditions separately.

For example:

- Stable regimes can use smoother filtering
- High-vibration regimes can use stronger noise suppression
- Rotational transitions can use transition-specific control logic
- Anomaly/noise observations can be isolated for separate review
- Different regimes can use different thresholds or model parameters

## Running the Project

Run the combined clustering pipeline:

```powershell
python src\cluster_time_series_methods.py
```

Generate comparison plots:

```powershell
python src\plot_kmeans_hdbscan_time_series.py
```

Launch the interactive dashboard:

```powershell
python src\dash_cluster_app.py
```

The application normally runs at:

```text
http://127.0.0.1:8050/
```

## Interpretation

KMeans produces a fixed number of behavioural groups. Every modelled observation is assigned to one of those groups.

HDBSCAN produces density-based behavioural regimes and preserves a separate anomaly/noise group for observations that do not belong to a stable dense pattern.

The final regime labels are intended to support:

- Behaviour segmentation
- High-frequency data separation
- Denoising
- Control engineering
- Trip comparison
- Operational monitoring

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
