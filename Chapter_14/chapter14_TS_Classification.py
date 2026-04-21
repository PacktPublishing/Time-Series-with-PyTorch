# Extracted from chapter14_TS_Classification.qmd
# Do not edit the source .qmd file directly.

#| label: Libraries, file path, pallet and fig size
#| message: false
#| echo: false
#| eval: true

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import euclidean
import seaborn as sns

from datetime import datetime
from pathlib import Path
from IPython.display import display, HTML

from aeon.distances import dtw_distance, msm_distance, euclidean_distance, wdtw_distance

palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]

class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 8
    img_dim2 = 4

# ----------------------------------------------------------------------

#| label: Some helper functions
#| message: false
#| echo: false
#| eval: true

def load_m5_subset(data_folder: Path) -> pd.DataFrame:
    """
    Load and prepare the M5 subset data
    
    Parameters
    ----------
    data_folder : Path
        Path to the data directory
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with proper date formatting and index
    """
    # Construct filepath
    filepath = data_folder / 'M5_t20_ABC.csv'
    
    # Verify file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    # Read data
    df = pd.read_csv(filepath, index_col=0)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove last 28 days of each series
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-28]
    df = df_sorted.groupby('item_id').apply(remove_tail_values).reset_index(drop=True)
    
    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    return df


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze the dataset structure and provide summary statistics
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Get ABC distribution by unique products
    abc_dist = df.groupby('item_id')['ABC_class'].first().value_counts().to_dict()
    
    analysis = {
        'n_series': df['item_id'].nunique(),
       # 'n_stores': df['state_id'].nunique(),
        'departments':df['dept_id'].nunique(),
        ''
        'date_range': (df['date'].min(), df['date'].max()),
        'total_sales': df['sold'].sum(),
        'mean_price': df['sell_price'].mean(),
        'abc_distribution': abc_dist,
        'sales_by_class': df.groupby('ABC_class')['sold'].sum().to_dict(),
        'avg_price_by_class': df.groupby('ABC_class')['sell_price'].mean().to_dict()
    }
    
    return analysis


def plot_sales_analysis(series: pd.DataFrame, figsize=(15, 10)):
    """
    Create a comprehensive sales analysis plot
    
    Parameters
    ----------
    series : pd.DataFrame
        Single time series data to analyze
    figsize : tuple
        Figure size for the plots
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Sales over time
    axes[0].plot(series['date'], series['sold'], color = darts_colors[1], alpha=0.7)
    axes[0].set_title(f"Sales Over Time for {series['item_id'].iloc[0]}")
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Units Sold')
    
    # Sales by day of week
    sns.boxplot(data=series, x='dayofweek', y='sold', ax=axes[1], color = darts_colors[2])
    axes[1].set_title('Sales Distribution by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Units Sold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(
        "\nSummary Statistics: "
        f"{series['sold'].describe()}"
    )

def plot_time_series(t, series_list, labels, title="Time Series Comparison"):
    """Plot multiple time series for comparison"""
    plt.figure(figsize=(10, 6))
    
    for i, series in enumerate(series_list):
        plt.plot(t, series, label=labels[i], color=palette[i % len(palette)])
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_distances(s1, s2):
    """Calculate various distance measures between two time series"""
    # Euclidean distance
    ed = euclidean_distance(s1, s2)
    
    # Dynamic Time Warping distance (no window constraint)
    dtw_dist = dtw_distance(s1, s2, window=None)
    
    # DTW with window constraint (Sakoe-Chiba band)
    # Window must be between 0 and 1 in aeon, representing the fraction of series length
    dtw_window1 = dtw_distance(s1, s2, window=0.05)  # 5% of series length
    dtw_window2 = dtw_distance(s1, s2, window=0.10)  # 10% of series length
    
    # Add weighted DTW for comparison
    wdtw_dist = wdtw_distance(s1, s2, g=0.1)  # g controls the penalty weight
    
    return {
        "Euclidean": ed,
        "DTW": dtw_dist,
        "DTW (window=5%)": dtw_window1,
        "DTW (window=10%)": dtw_window2,
        "WDTW": wdtw_dist
    }

def visualize_euclidean_matching(t, s1, s2, title="Euclidean Distance: Point-to-Point Alignment"):
    """Visualize how Euclidean distance aligns points at same time steps"""
    plt.figure(figsize=(12, 6))
    
    # Plot the two series
    plt.plot(t, s1, label="Series 1", color=palette[0])
    plt.plot(t, s2, label="Series 2", color=palette[1])
    
    # Draw lines between matching points (every 10th point for clarity)
    for i in range(0, len(t), 10):
        plt.plot([t[i], t[i]], [s1[i], s2[i]], 'k--', alpha=0.5)
        
    # Highlight a few matched points
    indices = [10, 30, 50, 70, 90]
    for i in indices:
        plt.scatter(t[i], s1[i], color='red', s=50, zorder=5)
        plt.scatter(t[i], s2[i], color='red', s=50, zorder=5)
    
    # Calculate and display the Euclidean distance
    ed = euclidean_distance(s1, s2)
    plt.text(0.02, 0.95, f"Euclidean Distance: {ed:.2f}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_dtw_matching(t, s1, s2, window=None, title="DTW: Optimal Path Alignment"):
    """Visualize how DTW aligns points across time using aeon"""
    # We need to compute DTW path ourselves since aeon doesn't expose alignment_path in the public API
    # First calculate the cost matrix and optimal path
    from scipy.spatial.distance import cdist
    
    # Create time point matrices
    s1_reshaped = s1.reshape(-1, 1)
    s2_reshaped = s2.reshape(-1, 1)
    
    # Calculate the cost matrix (pairwise Euclidean distances)
    cost_matrix = cdist(s1_reshaped, s2_reshaped, 'euclidean')
    
    # Initialize accumulated cost matrix
    acc_cost = np.zeros_like(cost_matrix)
    acc_cost[0, 0] = cost_matrix[0, 0]
    
    # Fill the first row and column
    for i in range(1, len(s1)):
        acc_cost[i, 0] = acc_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, len(s2)):
        acc_cost[0, j] = acc_cost[0, j-1] + cost_matrix[0, j]
    
    # Apply window constraint if specified
    if window is not None:
        # Convert window (fraction) to number of points
        if isinstance(window, float) and 0 <= window <= 1:
            window_size = int(window * len(s1))
        else:
            window_size = window
            
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                if abs(i - j) <= window_size:
                    acc_cost[i, j] = cost_matrix[i, j] + min(
                        acc_cost[i-1, j],      # insertion
                        acc_cost[i, j-1],      # deletion
                        acc_cost[i-1, j-1]     # match
                    )
                else:
                    acc_cost[i, j] = np.inf
    else:
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                acc_cost[i, j] = cost_matrix[i, j] + min(
                    acc_cost[i-1, j],      # insertion
                    acc_cost[i, j-1],      # deletion
                    acc_cost[i-1, j-1]     # match
                )
    
    # Backtrack to find the path
    path = []
    i, j = len(s1) - 1, len(s2) - 1
    path.append((i, j))
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1]])
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    
    path.reverse()
    alignment_pairs = path
    
    plt.figure(figsize=(12, 6))
    
    # Plot the two series
    plt.plot(t, s1, label="Series 1", color=palette[0])
    plt.plot(t, s2, label="Series 2", color=palette[1])
    
    # Draw lines between matching points along the DTW path
    # Use only a subset of the path for visualization clarity
    viz_path = alignment_pairs[::max(1, len(alignment_pairs)//15)]  # Sample points for clearer visualization
    
    for i, j in viz_path:
        plt.plot([t[i], t[j]], [s1[i], s2[j]], 'k--', alpha=0.5)
        
    # Highlight a few matched points
    highlight_indices = viz_path[1:-1:2]  # Skip first and last, take every 2nd
    for i, j in highlight_indices:
        plt.scatter(t[i], s1[i], color='red', s=50, zorder=5)
        plt.scatter(t[j], s2[j], color='red', s=50, zorder=5)
    
    # Calculate and display the DTW distance
    if window is not None:
        # Need to convert window to fraction for aeon
        if not isinstance(window, float) or window > 1:
            window_frac = min(1.0, window / len(s1))
        else:
            window_frac = window
        dtw_dist = dtw_distance(s1, s2, window=window_frac)
    else:
        dtw_dist = dtw_distance(s1, s2, window=None)
        
    plt.text(0.02, 0.95, f"DTW Distance: {dtw_dist:.2f}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    if window is not None:
        if isinstance(window, float) and window <= 1:
            title = f"{title} (Window={window*100:.0f}%)"
        else:
            title = f"{title} (Window={window} points)"
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_dtw_matrix(s1, s2, window=None, title="DTW Accumulated Cost Matrix"):
    """Visualize the DTW cost matrix and optimal path using numpy and scipy"""
    from scipy.spatial.distance import cdist
    
    # Create the cost matrix manually since we don't have access to aeon's internal functions
    s1_reshaped = s1.reshape(-1, 1)
    s2_reshaped = s2.reshape(-1, 1)
    
    # Calculate the cost matrix (pairwise Euclidean distances)
    cost_matrix = cdist(s1_reshaped, s2_reshaped, 'euclidean')
    
    # Initialize accumulated cost matrix
    acc_cost = np.zeros_like(cost_matrix)
    acc_cost[0, 0] = cost_matrix[0, 0]
    
    # Fill the first row and column
    for i in range(1, len(s1)):
        acc_cost[i, 0] = acc_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, len(s2)):
        acc_cost[0, j] = acc_cost[0, j-1] + cost_matrix[0, j]
    
    # Apply window constraint if specified
    if window is not None:
        # Convert window (fraction) to number of points
        if isinstance(window, float) and 0 <= window <= 1:
            window_size = int(window * len(s1))
        else:
            window_size = window
            
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                if abs(i - j) <= window_size:
                    acc_cost[i, j] = cost_matrix[i, j] + min(
                        acc_cost[i-1, j],      # insertion
                        acc_cost[i, j-1],      # deletion
                        acc_cost[i-1, j-1]     # match
                    )
                else:
                    acc_cost[i, j] = np.inf
    else:
        for i in range(1, len(s1)):
            for j in range(1, len(s2)):
                acc_cost[i, j] = cost_matrix[i, j] + min(
                    acc_cost[i-1, j],      # insertion
                    acc_cost[i, j-1],      # deletion
                    acc_cost[i-1, j-1]     # match
                )
    
    # Backtrack to find the path
    path = []
    i, j = len(s1) - 1, len(s2) - 1
    path.append((i, j))
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1]])
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    
    path.reverse()
    
    # Extract path coordinates
    path_y, path_x = zip(*path)
    
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap (blue to red)
    colors = [(0.0, palette[1]), (1.0, palette[6])]
    cm = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Make a copy of acc_cost for display (replace inf with max value)
    display_cost = acc_cost.copy()
    if window is not None:
        display_cost[display_cost == np.inf] = np.nanmax(display_cost[display_cost != np.inf])
    
    # Plot the accumulated cost matrix
    plt.imshow(display_cost, origin='lower', cmap=cm, aspect='auto')
    plt.colorbar(label='Accumulated Cost')
    
    # Plot the optimal warping path
    plt.plot(path_x, path_y, color='white', linewidth=2)
    
    # Calculate and display the DTW distance (the bottom-right value of cost matrix)
    # For window-constrained DTW, we should use aeon's function to get the true distance
    if window is not None:
        # Need to convert window to fraction for aeon
        if not isinstance(window, float) or window > 1:
            window_frac = min(1.0, window / len(s1))
        else:
            window_frac = window
        dtw_dist = dtw_distance(s1, s2, window=window_frac)
        
        if isinstance(window, float) and window <= 1:
            title = f"{title} (Window={window*100:.0f}%)"
        else:
            title = f"{title} (Window={window} points)"
    else:
        dtw_dist = dtw_distance(s1, s2, window=None)
    
    plt.text(0.02, 0.95, f"DTW Distance: {dtw_dist:.2f}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Series 2 Index", fontsize=12)
    plt.ylabel("Series 1 Index", fontsize=12)
    plt.tight_layout()
    plt.show()

def compare_distance_measures():
    """Run the complete demonstration of distance measures using aeon"""
    # Create example time series
    t, s1, s2, s3, s4, s5 = create_example_series()
    
    # 1. Visualize the different time series
    plot_time_series(
        t, 
        [s1, s2, s3, s4, s5], 
        ["Original", "Phase shifted", "Amplitude variation", "Different pattern", "Noisy signal"],
        "Sample Time Series for Distance Measure Comparison"
    )
    
    # 2. Euclidean vs DTW explanation
    # 2.1 Visualize Euclidean point-to-point matching
    visualize_euclidean_matching(t, s1, s2)
    
    # 2.2 Visualize DTW optimal alignment
    visualize_dtw_matching(t, s1, s2)
    
    # 3. DTW cost matrix and optimal path
    visualize_dtw_matrix(s1, s2)
    
    # 4. Impact of windowing constraint
    # Using window as percentage of series length (5%)
    window_size = int(0.05 * len(s1))
    visualize_dtw_matching(t, s1, s2, window=window_size)
    visualize_dtw_matrix(s1, s2, window=window_size)
    
    # 5. Compare actual distance values between original and each variant
    distances_table = []
    
    for i, s in enumerate([s2, s3, s4, s5]):
        name = ["Phase shifted", "Amplitude variation", "Different pattern", "Noisy signal"][i]
        distances = calculate_distances(s1, s)
        row = {"Series": name}
        row.update(distances)
        distances_table.append(row)
    
    df = pd.DataFrame(distances_table)
    print("\nDistance Comparison Table:")
    print(df.to_string(index=False))
    
    # Format as a bar chart for clearer comparison
    df_plot = df.set_index("Series")
    
    # Ensure we have enough colors for all distance measures
    bar_colors = palette[1:1+len(df_plot.columns)]
    
    ax = df_plot.plot(kind="bar", figsize=(14, 7), color=bar_colors)
    plt.title("Distance Measures Comparison", fontsize=14)
    plt.ylabel("Distance Value", fontsize=12)
    plt.legend(title="Distance Measure")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Add a comparison of aeon's other distance measures
    # Calculate MSM (Move-Split-Merge) distance
    msm_distances = []
    for i, s in enumerate([s2, s3, s4, s5]):
        name = ["Phase shifted", "Amplitude variation", "Different pattern", "Noisy signal"][i]
        msm_dist = msm_distance(s1, s, c=1.0)  # c is the cost parameter
        msm_distances.append({"Series": name, "MSM": msm_dist})
    
    msm_df = pd.DataFrame(msm_distances)
    print("\nMove-Split-Merge Distance Comparison:")
    print(msm_df.to_string(index=False))
    
    return df

# ----------------------------------------------------------------------

#| label: Creating some pseudo-data
#| message: false
#| echo: false
#| eval: true

t = np.linspace(0, 1, 100)
s1 = np.sin(2 * np.pi * t * 2)
s2 = np.sin(2 * np.pi * (t * 2 + 0.2))
s3 = 0.5 * np.sin(2 * np.pi * t * 2)
s4 = np.cos(2 * np.pi * t * 3)
s5 = np.sin(2 * np.pi * t * 2) + np.random.normal(0, 0.1, len(t))

# ----------------------------------------------------------------------

#| label: Euclidean distance of time series
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.1: Euclidean Distance: Point-to-Point Alignment" 
#| fig-cap-location: bottom
visualize_euclidean_matching(t, s1, s3, title=None)

# ----------------------------------------------------------------------

#| label: DTW accumulated cost matrix
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.2: DTW accumulated cost matrix" 
#| fig-cap-location: bottom
visualize_dtw_matrix(s1, s3, title=None)

# ----------------------------------------------------------------------

#| label: DTW of time series
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.3: DTW: Optimal Path Alignment" 
#| fig-cap-location: bottom
visualize_dtw_matching(t, s1, s3, title=None)

# ----------------------------------------------------------------------

#| label: Distance calculations 
#| message: false
#| echo: false
#| eval: true

distances = calculate_distances(s1, s3)
distances

# ----------------------------------------------------------------------

#| label: Loading data for TSC examples
#| message: false
#| echo: false
#| eval: true

# Load data
df = pd.read_csv('C:\\Users\\Graeme\\Documents\\github\\tsfwpt\\data\\dodgerloopday_imputed.csv')

# Split into train and test
train_df = df[df['source'] == 'train']
test_df = df[df['source'] == 'test']

print(df.head(n=3))

# ----------------------------------------------------------------------

#| label: Data prep for TSC
#| message: false
#| echo: false
#| eval: true


def prepare_data(df):
    series_list = []
    labels = []
    
    for series_id, group in df.groupby('series_id'):
        group = group.sort_values('time')
        series = group['value_filled'].values
        label = group['class'].iloc[0]
        series_list.append(series)
        labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(series_list)
    y = np.array(labels)
    
    # Reshape for aeon (samples, dimensions, timepoints)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)

# ----------------------------------------------------------------------

#| label: Comparing distances with KNN
#| message: false
#| echo: false
#| eval: true

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create and train KNN classifiers with different distance measures
classifiers = {
    "Euclidean": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean"),
    "Manhattan": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="manhattan"),
    "Minkowski": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="minkowski"),
    "DTW": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"),
    "WDTW": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="wdtw"),
    "MSM": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="msm")
}

results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name} classifier...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------

#| label: Plotting classification accuracy 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.4: Comparing accuracy of KNN classifcation with distance metrics" 
#| fig-cap-location: bottom

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=palette[:len(results)])
plt.ylabel("Accuracy", fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=14)

plt.xticks(fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

from aeon.classification.distance_based import ElasticEnsemble
ee = ElasticEnsemble(
    distance_measures=["euclidean","wdtw","ddtw","msm", "twe"],
    proportion_of_param_options=0.1,
    proportion_train_in_param_finding=0.3,
    proportion_train_for_test=0.1,
)
ee.fit(X_train, y_train)
ee_preds = ee.predict(X_test)
results["ElasticEnsemble"] = accuracy_score(y_test, ee_preds)

# ----------------------------------------------------------------------

#| label: Proximity forest default
#| message: false
#| echo: false
#| eval: true

from aeon.classification.distance_based import ProximityForest

start_time = time.time()
forest = ProximityForest(n_trees=20, n_splitters=5, max_depth=10)
forest.fit(X_train, y_train)
training_time = time.time() - start_time

forest_preds = forest.predict(X_test)
accuracy_score(y_test, forest_preds)
results["ProximityForest"] = accuracy_score(y_test, forest_preds)
print(f"Training time: {training_time:.2f} seconds") # 105.36 another time

# ----------------------------------------------------------------------

#| label: Proximity forest HP optimisation
#| message: false
#| echo: false
#| eval: false

import optuna

# def objective(trial):
#     # Define hyperparameters grid for optuna
#     n_trees = trial.suggest_int('n_trees', 10, 200)
#     n_splitters = trial.suggest_int('n_splitters', 2, 15)
#     max_depth = trial.suggest_int('max_depth', 3, 20)
    
#     # Create and train model with optuna trail HPs
#     forest = ProximityForest(
#         n_trees=n_trees,
#         n_splitters=n_splitters,
#         max_depth=max_depth
#     )
    
#     forest.fit(X_train, y_train)
#     predictions = forest.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
    
#     return accuracy  

# # Create a study object and optimize objective function
# study = optuna.create_study(direction='maximize') # because we want to tune for higher values
# study.optimize(objective, n_trials=20) 

# # Get best parameters
# best_params = study.best_params
# print(f"Best parameters: {best_params}")
# print(f"Best accuracy: {study.best_value:.4f}")

# ----------------------------------------------------------------------

#| label: Proximity forest tuned
#| message: false
#| echo: false
#| eval: false

best_params = {'n_trees': 104, 'n_splitters': 7, 'max_depth': 13} # taken from prior optuna run

# REQUIRES SEED: results vary 
# Train the final model with the best parameters
# start_time = time.time()
# best_forest = ProximityForest(
#     n_trees=best_params['n_trees'],
#     n_splitters=best_params['n_splitters'],
#     max_depth=best_params['max_depth']
# )

# best_forest.fit(X_train, y_train)
# training_time = time.time() - start_time
# best_predictions = best_forest.predict(X_test)
# best_accuracy = accuracy_score(y_test, best_predictions)

# Add to result dictionary
#results["ProximityForest (tuned)"] = best_accuracy
results["ProximityForest (tuned)"] = 0.675
# print(f"Training time: {training_time:.2f} seconds")

# ----------------------------------------------------------------------

#| label: Plotting all distance classifiers 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.5: Comparing accuracy of all distance classifiers" 
#| fig-cap-location: bottom

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=palette[:len(results)])
plt.ylabel("Accuracy", fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=14)

plt.xticks(fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Distrbution feature
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.5: Visualising distribution features" 
#| fig-cap-location: bottom

# Generate sample time series
np.random.seed(673)
t = np.linspace(0, 10, 500)
x = np.sin(t) + 0.5 * np.random.randn(len(t))

def visualize_distribution(x):
    """Visualize a time series and its distribution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the time series
    ax1.plot(x, color=palette[1])
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot the distribution (histogram)
    ax2.hist(x, bins=40, color=palette[2], alpha=0.7)
    ax2.axvline(np.mean(x), color=palette[6], linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(x):.2f}')
    ax2.axvline(np.median(x), color=palette[5], linestyle='--', 
                linewidth=2, label=f'Median: {np.median(x):.2f}')
    
    # Add variance annotation
    variance = np.var(x, ddof=1)
    ax2.text(0.05, 0.9, f'Variance: {variance:.2f}', transform=ax2.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_title('Distribution of Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

visualize_distribution(x)

# ----------------------------------------------------------------------

#| label: Stationarity Measures
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.6: Visualising stationarity measures" 
#| fig-cap-location: bottom

def statAv(x, window_size):
    """Measure of mean stationarity."""
    n = len(x)
    m = n // window_size
    window_means = [np.mean(x[i*window_size:(i+1)*window_size]) for i in range(m)]
    return np.std(window_means) / np.std(x)

def visualize_stationarity(x, window_size=50):
    """Visualize stationarity of a time series."""
    n = len(x)
    m = n // window_size
    
    # Calculate means in non-overlapping windows
    window_means = np.array([np.mean(x[i*window_size:(i+1)*window_size]) for i in range(m)])
    
    # Calculate standard deviations in non-overlapping windows
    window_stds = np.array([np.std(x[i*window_size:(i+1)*window_size], ddof=1) for i in range(m)])
    
    # Calculate StatAv
    stat_av = np.std(window_means) / np.std(x)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the original time series
    ax1.plot(x, color=palette[0], alpha=0.7)
    
    # Overlay the window means
    for i in range(m):
        start = i * window_size
        end = start + window_size
        if end > len(x):
            end = len(x)
        ax1.plot([start, end-1], [window_means[i], window_means[i]], 
                 color=palette[3], linewidth=2)
        ax1.axvline(start, color='gray', linestyle='--', alpha=0.3)
    
    ax1.set_title('Time Series with Window Means')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot window statistics
    ax2.plot(range(m), window_means, 'o-', color=palette[2], label='Window Means')
    ax2.plot(range(m), window_stds, 'o-', color=palette[4], label='Window StDevs')
    ax2.axhline(np.mean(x), color=palette[6], linestyle='--', 
                label=f'Overall Mean: {np.mean(x):.2f}')
    
    # Add StatAv annotation
    ax2.text(0.05, 0.9, f'StatAv: {stat_av:.2f}', transform=ax2.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_title('Window Statistics')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Statistic Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Create a non-stationary time series (with trend)
t = np.linspace(0, 10, 500)
non_stationary = np.sin(t) + 0.2 * t + 0.3 * np.random.randn(len(t))

visualize_stationarity(non_stationary)

# ----------------------------------------------------------------------

#| label: Autocorrelation Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.7: Visualising autocorrelation features" 
#| fig-cap-location: bottom
def autocorrelation(x, max_lag=50):
    """Compute autocorrelation for multiple lags."""
    n = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x, ddof=1)
    
    # Calculate autocorrelation for different lags
    ac = np.zeros(max_lag + 1)
    ac[0] = 1  # Autocorrelation with lag 0 is always 1
    
    for lag in range(1, max_lag + 1):
        for t in range(n - lag):
            ac[lag] += (x[t] - mean_x) * (x[t + lag] - mean_x)
        ac[lag] /= (n - lag) * var_x
    
    return ac

def visualize_autocorrelation(x, max_lag=50):
    """Visualize a time series and its autocorrelation function."""
    # Compute autocorrelation
    ac = autocorrelation(x, max_lag)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # plot
    ax1.plot(x, color=palette[1])
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    ax2.stem(range(max_lag + 1), ac, linefmt=palette[2], markerfmt=f'{palette[2]}', basefmt='gray')
    
    # Add confidence interval (95%)
    ci = 1.96 / np.sqrt(len(x))
    ax2.axhline(ci, color=palette[6], linestyle='--', alpha=0.7)
    ax2.axhline(-ci, color=palette[6], linestyle='--', alpha=0.7)
    ax2.fill_between(range(max_lag + 1), -ci, ci, color='gray', alpha=0.2)
    
    ax2.set_title('Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    plt.show()

# Create a time series with seasonality (periodic behavior)
t = np.linspace(0, 5, 500)
seasonal = 2 * np.sin(2 * np.pi * t) + np.random.randn(len(t)) * 0.5

visualize_autocorrelation(seasonal)

# ----------------------------------------------------------------------

#| label: Incremental Difference Features v1
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.8: High fluctuation feature" 
#| fig-cap-location: bottom

blue = '#0173B2' 
black = '#000000'  

# Set the style for better readability
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

# Example 1: Rossler-like with spikes (high_fluctuation example)
def create_spiky_series(n_points=1000, spike_interval=200, noise_level=0.1):
    """Create a time series with periodic spikes"""
    time = np.arange(n_points)
    signal = np.zeros(n_points)
    
    # Add periodic spikes
    for i in range(50, n_points, spike_interval):
        if i < n_points:
            signal[i-5:i+5] = np.exp(-np.abs(np.arange(-5, 5)) / 2)
            signal[i] = 2  # Peak value
    
    # Add small noise
    signal += np.random.normal(0, noise_level, n_points)
    
    return time, signal

# Example 2: Oscillating series with changing frequency (whiten_timescale example)
def create_oscillating_series(n_points=400):
    """Create an oscillating time series with moderate temporal structure"""
    time = np.arange(n_points)
    
    # Base oscillation with varying frequency
    frequency = 0.15 + 0.05 * np.sin(time * 0.01)  # Slightly varying frequency
    base_signal = 10 * np.sin(time * frequency)
    
    # Add a moderate trend component
    trend = 3 * np.sin(time * 0.02)
    
    # Add some structured noise (AR(1) process)
    noise = np.zeros(n_points)
    noise[0] = np.random.normal(0, 1)
    for i in range(1, n_points):
        noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 1.5)
    weight = 0.5 + 0.5 * (time > 200)  # Step change in the middle
    signal = base_signal + trend + weight * noise
    
    return time, signal

# Function to calculate incremental differences
def calc_incremental_differences(signal):
    """Calculate the incremental differences between consecutive points"""
    return np.diff(signal)

# Function to calculate high_fluctuation metric
def calc_high_fluctuation(signal, threshold_factor=0.04):
    """
    Calculate the proportion of incremental differences greater than
    threshold_factor * std of the signal
    """
    diffs = np.abs(calc_incremental_differences(signal))
    threshold = threshold_factor * np.std(signal)
    high_fluc = np.sum(diffs > threshold) / len(diffs)
    return high_fluc, diffs, threshold

# Function to calculate autocorrelation
def calc_autocorrelation(signal, max_lag=20):
    """Calculate autocorrelation up to max_lag"""
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    
    acf = np.zeros(max_lag + 1)
    acf[0] = 1  # Autocorrelation at lag 0 is 1
    
    for lag in range(1, max_lag + 1):
        numerator = np.sum((signal[lag:] - mean) * (signal[:n-lag] - mean))
        acf[lag] = numerator / ((n - lag) * var)
    
    return acf

# Function to find first zero crossing
def find_first_zero_crossing(acf):
    """Find the first zero crossing in autocorrelation function"""
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            # Linear interpolation to get more precise crossing
            if i > 0:
                x0, y0 = i-1, acf[i-1]
                x1, y1 = i, acf[i]
                zero_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                return zero_cross
            return i
    return len(acf)  # No crossing found

# Function to calculate whiten_timescale
def calc_whiten_timescale(signal, max_lag=20):
    """
    Calculate the ratio of first zero-crossing of the autocorrelation
    of the original signal vs. its incremental differences
    """
    # Original signal autocorrelation
    acf_orig = calc_autocorrelation(signal, max_lag)
    tau_orig = find_first_zero_crossing(acf_orig)
    
    # Incremental differences autocorrelation
    diffs = calc_incremental_differences(signal)
    acf_diffs = calc_autocorrelation(diffs, max_lag)
    tau_diffs = find_first_zero_crossing(acf_diffs)
    
    # Ratio (whiten_timescale)
    if tau_orig > 0:
        ratio = tau_diffs / tau_orig
    else:
        ratio = np.nan
        
    return ratio, acf_orig, acf_diffs, tau_orig, tau_diffs

# Create and plot Example 1 (high_fluctuation)
def plot_high_fluctuation_example():
    time, signal = create_spiky_series()
    high_fluc, diffs, threshold = calc_high_fluctuation(signal)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot the time series
    ax1.plot(time, signal, color=black, linewidth=1.5)
    ax1.set_title(f'Spiky Time Series [high_fluctuation = {high_fluc:.3f}]')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Time')
    
    # Plot the histogram of differences
    bins = np.linspace(0, 2, 50)
    ax2.hist(np.abs(diffs)/np.std(signal), bins=bins, color=blue, alpha=0.7, 
             edgecolor='white', linewidth=0.5)
    ax2.axvline(threshold/np.std(signal), color='red', linestyle='--', 
                linewidth=1.5, label=f'Threshold (0.04σ)')
    ax2.set_xlabel('Incremental differences (abs sigma)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Create and plot Example 2 (whiten_timescale)
def plot_whiten_timescale_example():
    time, signal = create_oscillating_series()
    
    # Calculate differences
    diffs = calc_incremental_differences(signal)
    time_diffs = time[1:]
    
    # Calculate whiten_timescale and autocorrelations
    ratio, acf_orig, acf_diffs, tau_orig, tau_diffs = calc_whiten_timescale(signal)
    
    # Create time lag vector for ACF plots
    lags = np.arange(len(acf_orig))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot the time series and its differences
    ax1.plot(time, signal, color=black, linewidth=1.5, label='Original')
    ax1.plot(time_diffs, diffs, color=blue, linewidth=1, alpha=0.7, linestyle='--', label='Differences')
    ax1.set_title(f'Oscillating Time Series [whiten_timescale = {ratio:.3f}]')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Time')
    ax1.legend()
    
    # Plot the autocorrelation functions
    ax2.plot(lags, acf_orig, color=black, linewidth=1.5, marker='o', markersize=5, label='Original ACF')
    ax2.plot(lags, acf_diffs, color=blue, linewidth=1.5, marker='^', markersize=5, label='Differences ACF')
    
    # Mark the zero crossings
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.plot(tau_orig, 0, 'ro', markersize=8, label=f'First Zero-Crossing (orig): {tau_orig:.1f}')
    ax2.plot(tau_diffs, 0, 'go', markersize=8, label=f'First Zero-Crossing (diff): {tau_diffs:.1f}')
    
    ax2.set_title(f'Autocorrelation Functions')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_xlabel('Time lag, τ')
    ax2.set_ylim(-0.7, 1.1)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Generate and display the plots
fig1 = plot_high_fluctuation_example()
plt.show()

# ----------------------------------------------------------------------

#| label: Incremental Difference Features v2
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.9: Whitening timescale feature" 
#| fig-cap-location: bottom
fig2 = plot_whiten_timescale_example()
plt.show()

# ----------------------------------------------------------------------

#| label: Forecasting Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.10: Forecast error feature" 
#| fig-cap-location: bottom
# Define colors

from scipy import stats

# Define colors
blue = '#0173B2'  
black = '#000000'  

# Set the style for better readability
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 16

# Create a simple time series with two predictability regimes
np.random.seed(59028)
t = np.arange(100)

# Create more predictable region
predictable = np.sin(t[:50] * 0.3) + np.random.normal(0, 0.2, 50)

# Create less predictable region (same frequency but more noise)
less_predictable = np.sin(t[50:] * 0.3) + np.random.normal(0, 0.8, 50)

# Combine both regions
time_series = np.concatenate([predictable, less_predictable])

# Z-score the time series
time_series = stats.zscore(time_series)

# Create 3-point rolling mean forecast
forecasts = np.zeros_like(time_series)
forecasts[:3] = np.nan  # First three points cannot be forecast with 3-point mean

for i in range(3, len(time_series)):
    forecasts[i] = np.mean(time_series[i-3:i])  # Mean of previous 3 values

# Calculate residuals (forecast errors)
residuals = time_series[3:] - forecasts[3:]

# Calculate forecast_error metrics for each region
forecast_error_full = np.std(residuals)
forecast_error_region1 = np.std(residuals[:47])  # First region (minus first 3 points)
forecast_error_region2 = np.std(residuals[47:])  # Second region

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot time series and forecasts
ax1.plot(t, time_series, color=black, linewidth=1.5, label='Original Time Series')
ax1.plot(t[3:], forecasts[3:], color=blue, linewidth=1.5, label='3-Point Rolling Mean Forecast')
ax1.set_xlabel('Time', fontsize=16)
ax1.set_ylabel('Value (z-scored)', fontsize=16)
ax1.legend(fontsize=16)

# Highlight the transition point
ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.text(30, max(time_series)+0.3, 'Lower Noise Region', fontsize=16, ha='center')
ax1.text(75, max(time_series)+0.3, 'Higher Noise Region', fontsize=16, ha='center')

# Plot residuals
ax2.plot(t[3:], residuals, color=blue, linewidth=1.5)
ax2.axhline(y=0, color=black, linestyle='-', alpha=0.3, linewidth=1.5)
ax2.fill_between(t[3:], 0, residuals, color=blue, alpha=0.2)
ax2.set_xlabel('Time', fontsize=16)
ax2.set_ylabel('Forecast Error', fontsize=16)
ax2.set_title(f'Forecast Residuals', fontsize=18)

# Add annotations for error metrics
ax2.text(25, max(residuals)-0.2, f'Region 1 Error = {forecast_error_region1:.3f}', 
         fontsize=16, ha='center', bbox=dict(facecolor='white', alpha=0.8))
ax2.text(75, max(residuals)-0.2, f'Region 2 Error = {forecast_error_region2:.3f}', 
         fontsize=16, ha='center', bbox=dict(facecolor='white', alpha=0.8))
ax2.text(50, min(residuals)+0.3, f'Full Series Error = {forecast_error_full:.3f}', 
         fontsize=16, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Reference line for std=1 (original time series std)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.text(95, 1.1, 'σ = 1', color='red', fontsize=16, ha='right')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Entropy Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.11: Visualising Entropy features" 
#| fig-cap-location: bottom
def sample_entropy(x, m=2, r=0.2):
    """Simplified implementation of Sample Entropy (SampEn)."""
    n = len(x)
    r = r * np.std(x, ddof=1)  # Convert r to absolute distance
    
    # Count similar patterns of length m and m+1
    def count_matches(template, patterns, r):
        return sum(1 for pattern in patterns if max(abs(template - pattern)) <= r)
    
    # Create embedding vectors of length m and m+1
    def create_vectors(m):
        return np.array([x[i:i+m] for i in range(n-m+1)])
    
    vectors_m = create_vectors(m)
    vectors_m1 = create_vectors(m+1)
    
    # Count matches (excluding self-matches)
    B = 0
    A = 0
    
    for i in range(n-m):
        template_m = vectors_m[i]
        template_m1 = vectors_m1[i]
        
        # Exclude self-match by comparing with patterns starting at different points
        similar_m = count_matches(template_m, 
                                  [vectors_m[j] for j in range(n-m+1) if abs(j-i) > 0], r)
        similar_m1 = count_matches(template_m1, 
                                   [vectors_m1[j] for j in range(n-m) if abs(j-i) > 0], r)
        
        B += similar_m / (n-m-1)
        A += similar_m1 / (n-m-1)
    
    # Average number of matches
    B /= (n-m)
    A /= (n-m)
    
    # Sample Entropy
    if A == 0 or B == 0:
        return float('inf')
    return -np.log(A/B)

def visualize_entropy(x, embedding_dims=range(1, 5)):
    """Visualize a time series and its sample entropy for different embedding dimensions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the time series
    ax1.plot(x, color=palette[1])
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Calculate and plot Sample Entropy for different embedding dimensions
    entropies = []
    for m in embedding_dims:
        se = sample_entropy(x, m=m)
        entropies.append(se)
    
    ax2.plot(embedding_dims, entropies, 'o-', color=palette[2], linewidth=2)
    ax2.set_title('Sample Entropy for Different Embedding Dimensions')
    ax2.set_xlabel('Embedding Dimension (m)')
    ax2.set_ylabel('Sample Entropy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generate different types of time series
np.random.seed(78447)
n = 200

# Random noise (high entropy)
random_noise = np.random.randn(n)

# Sine wave (low entropy, predictable)
t = np.linspace(0, 4*np.pi, n)
sine_wave = np.sin(t)

# Chaotic system (logistic map, moderate entropy)
chaotic = np.zeros(n)
chaotic[0] = 0.5
r = 3.9  # Chaotic regime
for i in range(1, n):
    chaotic[i] = r * chaotic[i-1] * (1 - chaotic[i-1])

visualize_entropy(chaotic)

# ----------------------------------------------------------------------

#| label: Fourier Transform Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.12: Visualising ourier features" 
#| fig-cap-location: bottom
def visualize_fourier(x, sampling_rate=1.0):
    """Visualize a time series and its frequency components."""
    n = len(x)
    
    # Compute the FFT
    fft_vals = np.fft.fft(x) / np.sqrt(n)
    fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Only plot the positive frequencies (up to Nyquist frequency)
    pos_freq_idx = fft_freq > 0
    freqs = fft_freq[pos_freq_idx]
    magnitudes = np.abs(fft_vals[pos_freq_idx])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the time series
    ax1.plot(np.arange(n)/sampling_rate, x, color=palette[1])
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # Plot the frequency spectrum
    ax2.stem(freqs, magnitudes, linefmt=palette[2], markerfmt=f'{palette[2]}', basefmt='gray')
    ax2.set_title('Frequency Spectrum (Fourier Transform)')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, sampling_rate/2)  # Limit to Nyquist frequency
    
    # Identify dominant frequencies
    top_k = 3
    dominant_idx = np.argsort(magnitudes)[-top_k:]
    for i, idx in enumerate(dominant_idx):
        freq = freqs[idx]
        mag = magnitudes[idx]
        ax2.annotate(f'{freq:.2f} Hz', 
                    xy=(freq, mag), 
                    xytext=(freq, mag + 0.5),
                    color=palette[i+4],
                    arrowprops=dict(facecolor=palette[i+4], shrink=0.05),
                    horizontalalignment='center')
    
    plt.tight_layout()
    plt.show()

# Create a time series with multiple frequency components
t = np.linspace(0, 2, 500)
multi_freq = 3*np.sin(2*np.pi*2*t) + 1.5*np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t) + 0.5*np.random.randn(len(t))

visualize_fourier(multi_freq, sampling_rate=250)

# ----------------------------------------------------------------------

#| label: Detrended Fluctuation Analysis (DFA)
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.13: Visualising detrended fluctuation analysis " 
#| fig-cap-location: bottom
def dfa(x, scales=None):
    """Compute Detrended Fluctuation Analysis."""
    if scales is None:
        scales = np.logspace(1, 2.6, 20).astype(int)
        scales = np.unique(scales)
    
    # Integrate the time series (cumulative sum of deviations from the mean)
    y = np.cumsum(x - np.mean(x))
    
    # Calculate fluctuation for different scales
    fluctuations = []
    for scale in scales:
        # Split into non-overlapping segments
        n_segments = len(y) // scale
        fluct = 0
        
        for i in range(n_segments):
            segment = y[i*scale:(i+1)*scale]
            # Fit a polynomial trend
            time_index = np.arange(len(segment))
            coeffs = np.polyfit(time_index, segment, 1)
            trend = np.polyval(coeffs, time_index)
            # Calculate variance of detrended segment
            fluct += np.sum((segment - trend)**2)
        
        # Mean fluctuation over all segments
        fluctuations.append(np.sqrt(fluct / (n_segments * scale)))
    
    # Calculate scaling exponent (alpha) from log-log plot
    coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
    alpha = coeffs[0]
    
    return scales, fluctuations, alpha

def visualize_dfa(x):
    """Visualize DFA analysis of a time series."""
    scales, fluctuations, alpha = dfa(x)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the integrated time series
    y = np.cumsum(x - np.mean(x))
    ax1.plot(y, color=palette[1])
    
    # Show detrending on a sample scale
    scale = scales[len(scales)//2]
    segment_start = 0
    segment = y[segment_start:segment_start+scale]
    time_index = np.arange(len(segment))
    coeffs = np.polyfit(time_index, segment, 1)
    trend = np.polyval(coeffs, time_index)
    
    ax1.plot(range(segment_start, segment_start+scale), segment, color=palette[2], linewidth=2)
    ax1.plot(range(segment_start, segment_start+scale), trend, color=palette[6], linewidth=2, 
             linestyle='--')
    
    ax1.set_title('Integrated Time Series with Sample Detrending')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Integrated Value')
    
    # Plot the log-log relationship
    ax2.loglog(scales, fluctuations, 'o', color=palette[3])
    
    # Add the fitted line
    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)
    fit_line = np.exp(np.polyval([alpha, log_fluct[0] - alpha * log_scales[0]], np.log(scales)))
    ax2.loglog(scales, fit_line, color=palette[5], linewidth=2)
    
    ax2.text(0.05, 0.9, f'Scaling Exponent α = {alpha:.3f}', transform=ax2.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_title('Detrended Fluctuation Analysis')
    ax2.set_xlabel('Scale (log)')
    ax2.set_ylabel('Fluctuation (log)')
    
    plt.tight_layout()
    plt.show()

# Create time series with different scaling properties
np.random.seed(42)
n = 2000

# White noise (α ≈ 0.5)
white_noise = np.random.randn(n)

# Brownian motion / random walk (α ≈ 1.5)
brownian = np.cumsum(np.random.randn(n))

# Long-range correlated noise (using fractional Gaussian noise approximation)
def fgn(n, H=0.8):
    """Generate fractional Gaussian noise with Hurst exponent H."""
    # Generate fractional Brownian motion
    t = np.arange(n+1)
    fbm = np.zeros(n+1)
    
    # Simple approximation
    for i in range(1, n+1):
        fbm[i] = fbm[i-1] + np.random.randn() * (i**H - (i-1)**H)
    
    # Convert to increments (fractional Gaussian noise)
    return np.diff(fbm)

long_range_correlated = fgn(n, H=0.8)

visualize_dfa(long_range_correlated)

# ----------------------------------------------------------------------

#| label: Complexity Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.14: Outlier clustering" 
#| fig-cap-location: bottom
# Define colors
blue = '#0173B2'  # Colorblind-friendly blue
black = '#000000'  # Black for contrast
red = '#D55E00'    # For positive outliers
green = '#009E73'  # For negative outliers

# Set the style for better readability
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 16

def calculate_outlier_timing(x, direction='pos', threshold_pct=0.01):
    """
    Calculate the timing of outliers in a time series.
    
    Parameters:
    -----------
    x : array-like
        Input time series (will be z-scored internally)
    direction : str, 'pos' or 'neg'
        Whether to look at positive or negative outliers
    threshold_pct : float between 0 and 1
        Percentage of the range to use for threshold spacing
        
    Returns:
    --------
    metric : float
        The outlier timing metric, between -1 (outliers at start) and 1 (outliers at end)
    thresholds : array-like
        The thresholds used
    rmds : array-like
        The rescaled median positions for each threshold
    """
    # Z-score the time series
    x_z = stats.zscore(x)
    n = len(x_z)
    
    # Create thresholds
    if direction == 'pos':
        max_val = np.max(x_z)
        thresholds = np.linspace(0, max_val, int(1/threshold_pct))
    else:  # 'neg'
        min_val = np.min(x_z)
        thresholds = np.linspace(0, min_val, int(1/threshold_pct))
        
    # Calculate the rescaled median position for each threshold
    rmds = []
    for threshold in thresholds:
        if direction == 'pos':
            over_threshold_indices = np.where(x_z > threshold)[0]
        else:  # 'neg'
            over_threshold_indices = np.where(x_z < threshold)[0]
            
        if len(over_threshold_indices) > 0:
            # Calculate the median position
            median_idx = np.median(over_threshold_indices)
            
            # Rescale to [-1, 1] range
            # -1: start of series, 0: middle, 1: end
            rmd = 2 * (median_idx / (n - 1)) - 1
            rmds.append(rmd)
        else:
            # No points over threshold
            continue
    
    # Median of all rmd values
    if rmds:
        metric = np.median(rmds)
    else:
        metric = np.nan
        
    return metric, thresholds[:len(rmds)], np.array(rmds)

def generate_time_series_with_outliers(n=400, outlier_position='start', n_outliers=20, magnitude=5):
    """
    Generate a time series with outliers at specific positions.
    
    Parameters:
    -----------
    n : int
        Length of the time series
    outlier_position : str, 'start', 'middle', 'end', or 'random'
        Where to place the outliers
    n_outliers : int
        Number of outliers to place
    magnitude : float
        Magnitude of outliers (in standard deviations)
        
    Returns:
    --------
    x : array-like
        The generated time series
    outlier_indices : array-like
        Indices where outliers were placed
    """
    # Generate base time series with AR(1) process
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = 0.7 * x[i-1] + 0.3 * np.random.randn()
    
    # Generate outlier indices based on position
    if outlier_position == 'start':
        outlier_indices = np.random.choice(np.arange(n//5), n_outliers, replace=False)
    elif outlier_position == 'middle':
        outlier_indices = np.random.choice(np.arange(2*n//5, 3*n//5), n_outliers, replace=False)
    elif outlier_position == 'end':
        outlier_indices = np.random.choice(np.arange(4*n//5, n), n_outliers, replace=False)
    else:  # 'random'
        outlier_indices = np.random.choice(np.arange(n), n_outliers, replace=False)
    
    # Add outliers
    outlier_signs = np.random.choice([-1, 1], n_outliers)
    for i, idx in enumerate(outlier_indices):
        x[idx] += outlier_signs[i] * magnitude
    
    return x, outlier_indices

def visualize_outlier_timing(x, title="Time Series with Outliers"):
    """Visualize a time series and its outlier timing metrics."""
    # Calculate metrics
    pos_metric, pos_thresholds, pos_rmds = calculate_outlier_timing(x, 'pos')
    neg_metric, neg_thresholds, neg_rmds = calculate_outlier_timing(x, 'neg')
    
    # Z-score the time series
    x_z = stats.zscore(x)
    n = len(x_z)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot time series with outliers highlighted
    ax1.plot(x_z, color=black, linewidth=1.5)
    
    # Highlight positive outliers
    pos_threshold = np.percentile(x_z, 95)  # Top 5% as outliers
    pos_outlier_indices = np.where(x_z > pos_threshold)[0]
    ax1.scatter(pos_outlier_indices, x_z[pos_outlier_indices], color=red, s=100, zorder=10, 
               label='Positive Outliers')
    
    # Highlight negative outliers
    neg_threshold = np.percentile(x_z, 5)  # Bottom 5% as outliers
    neg_outlier_indices = np.where(x_z < neg_threshold)[0]
    ax1.scatter(neg_outlier_indices, x_z[neg_outlier_indices], color=green, s=100, zorder=10,
               label='Negative Outliers')
    
    # Add thresholds
    ax1.axhline(y=pos_threshold, color=red, linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=neg_threshold, color=green, linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel('Value (z-scored)', fontsize=16)
    ax1.set_title(title, fontsize=18)
    ax1.legend(fontsize=16)
    
    # Plot the RMD values for different thresholds
    ax2.plot(pos_thresholds, pos_rmds, 'o-', color=red, label='Positive Outliers')
    ax2.plot(neg_thresholds, neg_rmds, 'o-', color=green, label='Negative Outliers')
    
    # Add reference lines
    ax2.axhline(y=0, color=black, linestyle='-', alpha=0.3, linewidth=1.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add metric values
    ax2.text(0.05, 0.9, f'Positive Outlier Timing = {pos_metric:.3f}', transform=ax2.transAxes, 
            fontsize=14, color=red, bbox=dict(facecolor='white', alpha=0.8))
    ax2.text(0.05, 0.8, f'Negative Outlier Timing = {neg_metric:.3f}', transform=ax2.transAxes, 
            fontsize=14, color=green, bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Threshold Value', fontsize=16)
    ax2.set_ylabel('Rescaled Median Position', fontsize=16)
    ax2.set_title('Outlier Timing Analysis', fontsize=18)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=16)
    
    # Add interpretation
    if pos_metric < -0.3:
        pos_interpretation = "Positive outliers occur mostly near the start"
    elif pos_metric > 0.3:
        pos_interpretation = "Positive outliers occur mostly near the end"
    else:
        pos_interpretation = "Positive outliers are distributed throughout"
        
    if neg_metric < -0.3:
        neg_interpretation = "Negative outliers occur mostly near the start"
    elif neg_metric > 0.3:
        neg_interpretation = "Negative outliers occur mostly near the end"
    else:
        neg_interpretation = "Negative outliers are distributed throughout"
    
    fig.text(0.5, 0.01, f"{pos_interpretation}\n{neg_interpretation}", 
             ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    return pos_metric, neg_metric

# Create time series with different outlier patterns
np.random.seed(42)

# Example 2: Outliers at the end
ts_end, _ = generate_time_series_with_outliers(outlier_position='end')
pos_metric_end, neg_metric_end = visualize_outlier_timing(ts_end, "Time Series with Outliers at End")

# ----------------------------------------------------------------------

#| label: Shaplet Features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.15: Visualising Shaplets features" 
#| fig-cap-location: bottom
from scipy.spatial.distance import euclidean

# Set up the visualization style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 18

# Define colors
teal = '#0173B2'  # For time series
red = '#D55E00'   # For shapelets/patterns

# Function to generate synthetic time series data
def generate_time_series(n_points=100, pattern_type='class1', noise_level=0.2):
    np.random.seed(42)  # For reproducibility
    t = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, noise_level, n_points)
    
    # Base signal
    base = 0.5 * np.sin(t) + noise
    
    # Add class-specific patterns
    if pattern_type == 'class1':
        # Class 1: Has smooth peaks (shapelet pattern)
        pattern_positions = [10, 30, 50, 70, 90]
        for pos in pattern_positions:
            if pos < n_points - 10:
                base[pos:pos+10] += 0.8 * np.sin(np.linspace(0, np.pi, 10))
    else:
        # Class 2: Has sharp zigzag patterns
        pattern_positions = [10, 25, 40, 55, 70, 85]
        for pos in pattern_positions:
            if pos < n_points - 6:
                zigzag = np.array([0, 0.7, 0, 0.7, 0, 0.7])
                base[pos:pos+6] += zigzag
    
    return base

# Function to compute the minimum distance between a shapelet and a time series
def shapelet_distance(shapelet, time_series):
    shapelet_len = len(shapelet)
    min_dist = float('inf')
    min_position = 0
    
    # Slide the shapelet across the time series
    for i in range(len(time_series) - shapelet_len + 1):
        subsequence = time_series[i:i+shapelet_len]
        dist = euclidean(shapelet, subsequence)
        if dist < min_dist:
            min_dist = dist
            min_position = i
    
    return min_dist, min_position

# Generate data
n_points = 120
ts_class1 = generate_time_series(n_points, 'class1')
ts_class2 = generate_time_series(n_points, 'class2')

# Define a shapelet (a subsequence pattern characteristic of class 1)
# This would normally be learned from data, but we'll define it explicitly
shapelet_len = 10
shapelet = 0.8 * np.sin(np.linspace(0, np.pi, shapelet_len))

# Calculate distances and best matching positions
dist1, pos1 = shapelet_distance(shapelet, ts_class1)
dist2, pos2 = shapelet_distance(shapelet, ts_class2)

# Create a detailed visualization showing the actual minimum distance calculation
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

# Plot shapelet
ax1.plot(shapelet, 'o-', color=red, linewidth=2, markersize=6)
ax1.set_title('Shapelet', fontsize=18)
ax1.set_ylabel('Value', fontsize=18)
ax1.set_xticklabels([])

# Plot Class 1 with sliding window visualization
ax2.plot(ts_class1, color=teal, linewidth=1.5)
ax2.set_title('Class 1 - Sliding Window Matching', fontsize=18)
ax2.set_ylabel('Value', fontsize=18)
ax2.set_xticklabels([])

# Plot multiple potential matches
slide_positions = [10, 30, 50, 70]
distances = []

for i, pos in enumerate(slide_positions):
    if pos < n_points - shapelet_len:
        subsequence = np.arange(pos, pos+shapelet_len)
        dist = euclidean(shapelet, ts_class1[subsequence])
        distances.append(dist)
        
        line_style = '-' if pos == pos1 else '--'
        alpha = 1.0 if pos == pos1 else 0.5
        ax2.plot(subsequence, ts_class1[subsequence], 'o-', color=red, 
                 linewidth=2, markersize=6, alpha=alpha, linestyle=line_style)
        ax2.text(pos+5, min(ts_class1)-0.2-i*0.1, f'Distance: {dist:.3f}', 
                fontsize=12, color=red if pos == pos1 else 'darkred')

# Highlight the best match
subsequence1 = np.arange(pos1, pos1+shapelet_len)
ax2.plot(subsequence1, ts_class1[subsequence1], 'o-', color='darkred', 
         linewidth=3, markersize=8)
ax2.text(pos1+shapelet_len/2, min(ts_class1)-0.6, 'Best Match', 
        fontsize=14, color='darkred', ha='center', 
        bbox=dict(facecolor='white', alpha=0.8))

# Plot distance distribution
slide_range = range(n_points - shapelet_len + 1)
all_distances = []

for i in slide_range:
    subsequence = np.arange(i, i+shapelet_len)
    dist = euclidean(shapelet, ts_class1[subsequence])
    all_distances.append(dist)

ax3.plot(all_distances, color=teal, linewidth=1.5)
ax3.set_title('Distance to Shapelet at Each Position', fontsize=18)
ax3.set_xlabel('Starting Position of Subsequence', fontsize=18)
ax3.set_ylabel('Euclidean Distance', fontsize=18)

# Highlight minimum distance
ax3.scatter(pos1, dist1, color=red, s=100)
ax3.annotate(f'Minimum Distance: {dist1:.3f}', xy=(pos1, dist1), 
             xytext=(pos1+10, dist1+0.5), fontsize=14, color=red,
             arrowprops=dict(facecolor=red, shrink=0.05, width=1.5))

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Featurisation example
#| message: false
#| echo: true
#| eval: true

from aeon.transformations.collection.feature_based import Catch22

feature_transformer = Catch22(features = "all",n_jobs=10, replace_nans = True)
series_transformed = feature_transformer.fit_transform(X_train)

# ----------------------------------------------------------------------

#| label: Featurisation table
#| message: false
#| echo: true
#| eval: true

feature_names = [
    "DN_HistogramMode_5", "DN_HistogramMode_10", 
    "CO_f1ecac", "CO_FirstMin_ac", 
    "SB_BinaryStats_mean_longstretch1", 
    "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd", 
    "FC_LocalSimple_mean1_tauresrat", 
    "CO_trev_1_num", 
    "CO_HistogramAMI_even_2_5", 
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", 
    "MD_hrv_classic_pnn40", 
    "SB_BinaryStats_diff_longstretch0", 
    "SB_TransitionMatrix_3ac_sumdiagcov", 
    "PD_PeriodicityWang_th0_01", 
    "CO_Embed2_Dist_tau_d_expfit_meandiff", 
    "IN_AutoMutualInfoStats_40_gaussian_std", 
    "FC_LocalSimple_mean3_stderr", 
    "CO_HistogramAMI_even_10_5", 
    "CO_Embed2_Dist_tau_d_expfit_lambda", 
    "MD_hrv_classic_pnn20", 
    "SB_BinaryStats_mean_longstretch0"
]

# Create the dataframe with feature values
featurised_df = pd.DataFrame(series_transformed, columns=feature_names)

# Add series_id and labels to the dataframe
featurised_df.insert(0, 'series_id', range(len(series_transformed)))
featurised_df.insert(1, 'label', y_train)

# Apply styling with limited width
pd.set_option('display.max_colwidth', 20)  # Limit column width for display

# Create HTML with custom CSS for column width
html = """
<style>
    table {
        width: auto !important;
        table-layout: fixed;
    }
    th, td {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100px;
        padding: 5px !important;
    }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {
        max-width: 60px;
    }
</style>
"""
html += featurised_df.head(5).to_html(index=False, classes="table table-striped table-hover")

# Display the styled table
display(HTML(html))

# ----------------------------------------------------------------------

#| label: Featurisation catch22 classification
#| message: false
#| echo: true
#| eval: true
#| fig-cap: "14.16: catch22 classification" 
#| fig-cap-location: bottom

from aeon.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier

c22 = Catch22Classifier(
    estimator=RandomForestClassifier(
        n_estimators=291,      
        max_depth=5,           
        min_samples_split=12,  
        min_samples_leaf=5,    
        max_features='sqrt',   
        bootstrap=True,        
        class_weight='balanced', 
        random_state=3542
    ),
    features='all',
    catch24=False,             
    outlier_norm=True,         
    replace_nans=True,
    n_jobs=10,
    random_state=3542
)
c22.fit(X_train, y_train)
c22_preds = c22.predict(X_test)
results["catch22class"] = accuracy_score(y_test, c22_preds)

# ----------------------------------------------------------------------

# from aeon.classification.feature_based import FreshPRINCEClassifier
# fp = FreshPRINCEClassifier(default_fc_parameters="efficient", n_jobs=10, n_estimators = 200, verbose = -1, random_state = 866)
# fp.fit(X_train, y_train)
# fp_preds = c22cls.predict(X_test)
# results["freshprince"] = accuracy_score(y_test, c22_preds)

# ----------------------------------------------------------------------

#| label: Plotting all feature classifiers 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.17: Comparing accuracy of feature classifiers" 
#| fig-cap-location: bottom

selected_keys = ['Euclidean', 'WDTW', 'ProximityForest (tuned)', 'catch22class']
results_filtered = {k: results[k] for k in selected_keys if k in results}

plt.figure(figsize=(10, 6))
plt.bar(results_filtered.keys(), results_filtered.values(), color=palette[:len(results_filtered)])
plt.ylabel("Accuracy", fontsize=18)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results_filtered.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=18)

plt.xticks(fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Shaplet classification
#| message: false
#| echo: true
#| eval: true
from aeon.classification.shapelet_based import (
    ShapeletTransformClassifier,
    RDSTClassifier,
    SASTClassifier,
    LearningShapeletClassifier
)

# Shared parameters
RANDOM_STATE = 3542

# Common RandomForest parameters
rf_params = {
    'n_estimators': 300,
    'max_depth': 5,
    'min_samples_split': 10,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}

# Define classifiers
classifiers = {
    "ShapeletTransform": ShapeletTransformClassifier(
        estimator=RandomForestClassifier(**rf_params),
        max_shapelets=10,
        n_shapelet_samples=500,
        random_state=RANDOM_STATE
    ),
    "RDST": RDSTClassifier(
        # For RDST, we pass RF as the estimator instead of direct parameters
        estimator=RandomForestClassifier(**rf_params),
        random_state=RANDOM_STATE
    ),
    "SAST": SASTClassifier(
        classifier=RandomForestClassifier(**rf_params),
    ),
    # "LearningShapelet": LearningShapeletClassifier(
    #     n_shapelets_per_size=3,
    #     max_iter=1000,
    #     random_state=RANDOM_STATE
    # )
}

# Fit classifiers and evaluate
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

# ----------------------------------------------------------------------

#| label: Plotting all shapelet classifiers 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.18: Comparing accuracy of shapelet classifiers" 
#| fig-cap-location: bottom

selected_keys = ['Euclidean', 'WDTW', 'ProximityForest (tuned)', 'catch22class', 'ShapeletTransform', 'RDST','SAST']
results_filtered = {k: results[k] for k in selected_keys if k in results}

plt.figure(figsize=(10, 6))
plt.bar(results_filtered.keys(), results_filtered.values(), color=palette[:len(results_filtered)])
plt.ylabel("Accuracy", fontsize=18)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results_filtered.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=18)

plt.xticks(fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Dictionary apporach  
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.19: Diagram of dictionary apporach" 
#| fig-cap-location: bottom

from scipy import signal

np.random.seed(2542)

# Create a sample time series with noise
t = np.linspace(0, 1000, 1000)
sample = signal.chirp(t, f0=0.01, f1=0.05, t1=500, method='quadratic') * 0.5
envelope = np.exp(-0.5 * ((t - 500) / 150) ** 2)
sample = sample * envelope
# Add noise
sample += np.random.normal(0, 0.02, size=len(sample))

# Define window positions and size
window_size = 100
window_positions = [50, 200, 400, 600, 800]
windows = []
for pos in window_positions:
    if pos + window_size <= len(sample):
        windows.append(sample[pos:pos+window_size])

# Normalize windows
normalized_windows = []
for window in windows:
    mean = np.mean(window)
    std = np.std(window)
    if std > 0:
        normalized_windows.append((window - mean) / std)
    else:
        normalized_windows.append(window - mean)

# Create patterns with varying frequencies
patterns = ['bca', 'cab', 'bbb', 'bbc', 'abc', 'aaa', 'acc', 'bcb', 'ccc', 'cba']
pattern_counts = [3.2, 2.8, 2.5, 2.2, 2.1, 1.7, 1.4, 1.2, 1.2, 0.5]

# Plotting
plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 14})

# 1. Original time series
plt.subplot(4, 1, 1)
plt.plot(t, sample, color='#1f77b4', linewidth=1.5)
plt.title("Sample", fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 0.5)

# 2. Windowing
plt.subplot(4, 1, 2)
plt.title("(1) Windowing", fontsize=16)
plt.plot(t, sample, color='#1f77b4', alpha=0.15, linewidth=1)

for i, (window, pos) in enumerate(zip(normalized_windows, window_positions)):
    plt.plot(t[pos:pos+window_size], window, color='#1f77b4', linewidth=1.5)
    
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(-2.5, 2.5)

# 3. Discretization - matching the original layout
plt.subplot(4, 1, 3)
plt.title("(2) Discretization", fontsize=16)
plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.axis('off')  # Turn off axes

# Place "bcb" at specific position from original image
plt.text(100, 0.5, "bca", fontsize=14, color='#1f77b4')
plt.text(250, 0.5, "bbb", fontsize=14, color='#1f77b4')
plt.text(450, 0.5, "bbb", fontsize=14, color='#1f77b4')
plt.text(650, 0.5, "bbb", fontsize=14, color='#1f77b4')
plt.text(850, 0.5, "bbb", fontsize=14, color='#1f77b4')

# 4. Bag of Patterns histogram
plt.subplot(4, 1, 4)
plt.title("(3) Bag-of-Patterns model", fontsize=16)
plt.bar(range(len(patterns)), pattern_counts, color='#1f77b4')
plt.xticks(range(len(patterns)), patterns)
plt.ylabel("Counts")
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=1.5)
plt.show()

# ----------------------------------------------------------------------

#| label: Dictionary classification code  
#| message: false
#| echo: true
#| eval: true

# Dictionary-based classifier implementations
# from aeon.classification.dictionary_based import BOSSEnsemble, WEASEL_V2, TemporalDictionaryEnsemble

# # BOSS Ensemble
# boss = BOSSEnsemble(
#     max_ensemble_size=5, 
#     min_window = 60,
#     random_state=3542,
#     n_jobs=10
# )

# # WEASEL V2
# weasel = WEASEL_V2(
#     random_state=3542,
#     n_jobs=10
# )

# # Temporal Dictionary Ensemble
# tde = TemporalDictionaryEnsemble(
#     n_parameter_samples=250,   
#     max_ensemble_size=50,      
#     randomly_selected_params=True,
#     random_state=3542,
#     n_jobs=10
# )

# # Train and evaluate classifiers
# boss.fit(X_train, y_train)
# boss_preds = boss.predict(X_test)
# results["BOSS"] = accuracy_score(y_test, boss_preds)

# weasel.fit(X_train, y_train)
# weasel_preds = weasel.predict(X_test)
# results["WEASEL_V2"] = accuracy_score(y_test, weasel_preds)

# tde.fit(X_train, y_train)
# tde_preds = tde.predict(X_test)
# results["TDE"] = accuracy_score(y_test, tde_preds)

results["BOSS"] = 0.475
results["WEASEL_V2"] = 0.65
results["TDE"] = 0.55

# ----------------------------------------------------------------------

#| label: Plotting all dictionary classifiers 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.20: Comparing accuracy of dictionary classifiers" 
#| fig-cap-location: bottom

selected_keys = ['Euclidean', 'WDTW', 'ProximityForest (tuned)', 'catch22class', 'RDST','BOSS', 'WEASEL_V2','TDE']
results_filtered = {k: results[k] for k in selected_keys if k in results}

plt.figure(figsize=(10, 6))
plt.bar(results_filtered.keys(), results_filtered.values(), color=palette[:len(results_filtered)])
plt.ylabel("Accuracy", fontsize=18)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results_filtered.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=18)

plt.xticks(fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Kernel classification 
#| message: false
#| echo: true
#| eval: true

# ROCKET classifier implementation
from aeon.classification.convolution_based import RocketClassifier, MiniRocketClassifier
from sklearn.linear_model import RidgeClassifierCV

# Standard ROCKET
rocket = RocketClassifier(
    n_kernels=10000,          # Num random convolutional kernels
    estimator=RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 11)
    ),
    class_weight=None,
    n_jobs=10,    
    random_state=3542
)

# MiniROCKET variant - more parameter efficient
mini_rocket = MiniRocketClassifier(
    n_kernels = 10000, 
    max_dilations_per_kernel = 32, 
    estimator=RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 11)
    ),
    class_weight=None,
    random_state=3542
)

# Train and evaluate ROCKET
rocket.fit(X_train, y_train)
rocket_preds = rocket.predict(X_test)
results["ROCKET"] = accuracy_score(y_test, rocket_preds)

mini_rocket.fit(X_train, y_train)
mini_rocket_preds = mini_rocket.predict(X_test)
results["MiniROCKET"] = accuracy_score(y_test, mini_rocket_preds)

# ----------------------------------------------------------------------

#| label: Plotting kernel classifiers 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "14.22: Comparing accuracy of kernel classifiers" 
#| fig-cap-location: bottom

selected_keys = ['Euclidean', 'WDTW', 'ProximityForest (tuned)', 'catch22class', 'RDST', 'WEASEL_V2', 'ROCKET','MiniROCKET']
results_filtered = {k: results[k] for k in selected_keys if k in results}

plt.figure(figsize=(10, 6))
plt.bar(results_filtered.keys(), results_filtered.values(), color=palette[:len(results_filtered)])
plt.ylabel("Accuracy", fontsize=18)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, (method, accuracy) in enumerate(results_filtered.items()):
    plt.text(i, accuracy + 0.02, f"{accuracy:.4f}", ha='center', fontsize=18)

plt.xticks(fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: ResNet classification 
#| message: false
#| echo: true
#| eval: false
from aeon.classification.deep_learning import ResNetClassifier

# Some settings
n_epochs = 200         # Default is typically 1500 - reduced for time
batch_size = 64
random_state = 3542
n_jobs = 10
verbose = False

# ResNetClassifier
# Note. ResNet benefits from multiple residual blocks
resnet = ResNetClassifier(
    n_residual_blocks=3,              
    n_conv_per_residual_block=3,      
    n_filters=[128, 64, 64],          
    kernel_size=[8, 5, 3],            
    activation='relu',                
    n_epochs=n_epochs,
    batch_size=batch_size,
    random_state=random_state,
    verbose=verbose
)

resnet.fit(X_train, y_train)
resnet_preds = resnet.predict(X_test)
results["ResNet"] = accuracy_score(y_test, resnet_preds)

# ----------------------------------------------------------------------

#| label: InceptionTime classification 
#| message: false
#| echo: true
#| eval: false
from aeon.classification.deep_learning import InceptionTimeClassifier

inception_time = InceptionTimeClassifier(
    n_classifiers=3,
    n_epochs=n_epochs,
    batch_size=batch_size,
    random_state=random_state,
    verbose=verbose
)
inception_time.fit(X_train, y_train)
inception_time_preds = inception_time.predict(X_test)
results["InceptionTime"] = accuracy_score(y_test, inception_time_preds)

# ----------------------------------------------------------------------

#| label: H-InceptionTime classification 
#| message: false
#| echo: true
#| eval: false
from aeon.classification.deep_learning import IndividualInceptionClassifier

h_inception = IndividualInceptionClassifier(
    n_epochs=n_epochs,
    batch_size=batch_size,
    random_state=random_state,
    verbose=verbose
)
h_inception.fit(X_train, y_train)
h_inception_preds = h_inception.predict(X_test)
results["H-Inception"] = accuracy_score(y_test, h_inception_preds)

# ----------------------------------------------------------------------

#| label: LITETime classification 
#| message: false
#| echo: true
#| eval: false
from aeon.classification.deep_learning import LITETimeClassifier

lite_time = LITETimeClassifier(
    n_classifiers=3,
    n_epochs=n_epochs,
    batch_size=batch_size,
    random_state=random_state,
    verbose=verbose
)
lite_time.fit(X_train, y_train)
lite_time_preds = lite_time.predict(X_test)
results["LITETime"] = accuracy_score(y_test, lite_time_preds)