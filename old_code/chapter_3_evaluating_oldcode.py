# %% [markdown]
# # Time Series Analysis and Forecasting
# This notebook demonstrates fundamental concepts and techniques in time series analysis and forecasting.

# %%
#| label: Setup
#| echo: false
#| eval: true

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dateutil.relativedelta import relativedelta
from pathlib import Path 
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Global configuration
class CFG:
    # Try multiple approaches to find the data directory
    # Option 1: Check if data is in the repo structure (2 levels up from current directory)
    repo_data_path = Path.cwd().parent.parent / "data"
    # Option 2: Check if data is one level up (common structure)
    alt_data_path = Path.cwd().parent / "data"
    # Option 3: Check if data is in the current directory
    local_data_path = Path.cwd() / "data"
    
    # Try paths in order of likelihood
    if repo_data_path.exists():
        data_folder = repo_data_path
    elif alt_data_path.exists():
        data_folder = alt_data_path
    elif local_data_path.exists():
        data_folder = local_data_path
    else:
        # Fallback: Allow user to specify the path
        print("⚠️ Data directory not found! Please update CFG.data_folder with the correct path.")
        # Default to your current path as a starting point
        data_folder = Path('c:/Users/Graeme/Documents/github/tsfwpt/data')
        print(f"Currently set to: {data_folder}")
        print("Common usage patterns:")
        print("1. CFG.data_folder = Path('/absolute/path/to/data')")
        print("2. CFG.data_folder = Path.home() / 'github' / 'tsfwpt' / 'data'")
    
    img_dim1 = 8
    img_dim2 = 4
    
# Define a consistent color palette to use throughout the notebook
COLORS = {
    "black": "black",
    "blue": "#003DFD",
    "purple": "#b512b8",
    "teal": "#11a9ba",
    "green": "#0d780f",
    "orange": "#f77f07",
    "red": "#ba0f0f"
}

# Create a list version for easy indexing
COLOR_LIST = [COLORS["black"], COLORS["blue"], COLORS["purple"], 
              COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"]]

# %%
#| label: Plotting Functions
#| echo: false
#| eval: true

def darts_plot(df, x_column, y_columns, labels=None, quantiles=None, title=None, fontsize=18):
    """
    Creates a time series plot with confidence intervals similar to the darts library style.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis (typically dates)
        y_columns: List of column names to plot
        labels: Optional list of labels for the legend
        quantiles: Optional dict mapping column names to (low, high) quantile column names
        title: Optional plot title
        fontsize: Font size for all text elements
    """
    # Set the default font size for all text elements
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if labels is None:
        labels = y_columns
    
    alpha_confidence_intvls = 0.25

    for i, y_col in enumerate(y_columns):
        color = COLOR_LIST[i % len(COLOR_LIST)]
        
        # Plot the main line
        ax.plot(df[x_column], df[y_col], label=labels[i], color=color, linewidth=2)

        # Plot confidence interval if quantiles are provided
        if quantiles is not None and y_col in quantiles:
            low_col, high_col = quantiles[y_col]
            low_series = df[low_col]
            high_series = df[high_col]
            
            if len(low_series) > 1:
                ax.fill_between(
                    df[x_column],
                    low_series,
                    high_series,
                    color=color,
                    alpha=alpha_confidence_intvls,
                )
            else:
                ax.plot(
                    [df[x_column].iloc[0], df[x_column].iloc[0]],
                    [low_series.iloc[0], high_series.iloc[0]],
                    "-+",
                    color=color,
                    lw=2,
                )
    
    # Styling with specified font sizes
    ax.set_xlabel(x_column, fontsize=fontsize)
    ax.set_ylabel('Value', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure tick labels use the font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Create and style the legend
    legend = ax.legend(prop={'size': fontsize})
    
    # Set background color to white
    ax.set_facecolor('white')
    
    # Improve layout
    plt.tight_layout()
    
    return fig, ax

def plot_seasonal_decompose(series, model='additive', figsize=(12, 10), fontsize=18):
    """
    Plot the decomposition of a time series into trend, seasonal, and residual components.
    
    Args:
        series: Time series to decompose
        model: 'additive' or 'multiplicative'
        figsize: Figure size tuple
        fontsize: Font size for text elements
    """
    # Set global font size
    plt.rcParams.update({'font.size': fontsize})
    
    decomposition = seasonal_decompose(series, model=model)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Observed component
    decomposition.observed.plot(ax=ax1, color=COLORS["black"], linewidth=0.5)
    ax1.set_ylabel('Observed', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax1.set_title('Observed Time Series', fontsize=fontsize)
    
    # Trend component
    decomposition.trend.plot(ax=ax2, color=COLORS["blue"], linewidth=1)
    ax2.set_ylabel('Trend', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.set_title('Trend Component', fontsize=fontsize)
    
    # Seasonal component
    decomposition.seasonal.plot(ax=ax3, color=COLORS["purple"], linewidth=0.5)
    ax3.set_ylabel('Seasonal', fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax3.set_title('Seasonal Component', fontsize=fontsize)
    
    # Residual component
    ax4.scatter(decomposition.resid.index, decomposition.resid, color=COLORS["teal"], s=3, alpha=0.5)
    ax4.set_ylabel('Residual', fontsize=fontsize)
    ax4.set_xlabel('Date', fontsize=fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax4.set_title('Residual Component', fontsize=fontsize)
    
    # Improve layout with more space for the larger text
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    
    return fig

# Metrics calculation functions
def calculate_metrics(actual, forecast):
    """Calculate common error metrics for forecasting"""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual+1, np.array(forecast)+1)  # +1 to avoid division by zero
    return mae, rmse, mape

def calculate_scaled_metrics(actual, forecast, training_data):
    """Calculate scaled error metrics for forecasting"""
    in_sample_mae = np.mean(np.abs(np.diff(training_data)))
    if in_sample_mae == 0:
        return np.inf, np.inf
    errors = np.abs(actual - forecast)
    squared_errors = (actual - forecast) ** 2
    mase = np.mean(errors) / in_sample_mae
    rmsse = np.sqrt(np.mean(squared_errors)) / in_sample_mae
    return mase, rmsse

def calculate_relative_errors(actual, forecast, baseline_metrics):
    """Calculate relative error metrics compared to a baseline model"""
    mae, rmse, mape = calculate_metrics(actual, forecast)
    rel_mae = mae / baseline_metrics[0] if baseline_metrics[0] != 0 else np.inf
    rel_rmse = rmse / baseline_metrics[1] if baseline_metrics[1] != 0 else np.inf
    rel_mape = mape / baseline_metrics[2] if baseline_metrics[2] != 0 else np.inf
    return rel_mae, rel_rmse, rel_mape

# %%
#| label: Bias-variance tradeoff
#| echo: false
#| eval: true
#| fig-cap: 'Bias-variance tradeoff visualization'
#| fig-cap-location: bottom

# Generate data points
x = np.linspace(0, 10, 100)

# Define functions for bias, variance, and total error
bias = 10 * np.exp(-0.5 * x)
variance = 0.05 * (x ** 2)
total_error = bias + variance

# Create plot
plt.figure(figsize=(10, 6))

# Plot curves
plt.plot(x, bias, color=COLORS["black"], label='Bias²')
plt.plot(x, variance, color=COLORS["blue"], label='Variance', linestyle='dashdot')
plt.plot(x, total_error, color=COLORS["purple"], label='Total Error', linestyle=':')

# Vertical line for optimal complexity
optimal_x = (x[np.argmin(total_error)]) - 0.2
plt.axvline(x=optimal_x, color=COLORS["teal"], linestyle='--', label='Optimal Model Complexity')

# Labels and annotations
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend(fontsize=12)
plt.text(0.5, 9, 'Bias²', color=COLORS["black"], fontsize=12)
plt.text(8, 2.8, 'Variance', color=COLORS["blue"], fontsize=12)
plt.text(5, 3, 'Total Error', color=COLORS["purple"], fontsize=12)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# %%
#| label: Overfitting plot
#| echo: false
#| eval: true
#| fig-cap: 'Training and validation loss over iterations'
#| fig-cap-location: bottom

# Generate data points
iterations = np.linspace(0, 100, 1000)

# Define functions for training and validation loss
training_loss = 1.6 * np.exp(-0.03 * iterations) + 0.1
validation_loss = 1.6 * np.exp(-0.03 * iterations) + 0.1 + 0.0002 * iterations**1.8

# Find the minimum point of validation loss
overfitting_point = iterations[np.argmin(validation_loss)]

# Plotting
plt.figure(figsize=(10, 6))

# Plot curves
plt.plot(iterations, training_loss, color=COLORS["black"], label='Training loss', linewidth=2)
plt.plot(iterations, validation_loss, color=COLORS["blue"], label='Validation loss', linewidth=2)

# Vertical line at start of overfitting
plt.axvline(x=overfitting_point, color=COLORS["teal"], linestyle='--', linewidth=2)

# Annotation
plt.text(overfitting_point + 2, 1.2, 'Overfitting starts from here', 
         rotation=0, verticalalignment='center', fontsize=12, color=COLORS["teal"])

# Labels etc
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss over Iterations', fontsize=16)
plt.legend(fontsize=12)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust y-axis 
plt.ylim(0, 1.6)

plt.tight_layout()
plt.show()

# %%
#| label: Data wrangling
#| echo: true
#| eval: true

# Load data
try:
    panel_df = pd.read_csv(CFG.data_folder / 'M5_t20_ABC.csv', index_col=False)
    # Drop unnamed index column if present
    if 'Unnamed: 0' in panel_df.columns:
        panel_df = panel_df.drop(columns='Unnamed: 0')
    print(f"✓ Successfully loaded data from {CFG.data_folder / 'M5_t20_ABC.csv'}")
except FileNotFoundError:
    print(f"❌ Could not find the file: {CFG.data_folder / 'M5_t20_ABC.csv'}")
    print("Please check if the file exists or update CFG.data_folder.")
    # Create dummy data for demonstration if file not found
    print("Creating dummy data for demonstration...")
    import numpy as np
    dates = pd.date_range(start='2015-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Create 3 items with different patterns
    items = ['FOODS_2_197', 'FOODS_1_123', 'HOUSEHOLD_1_456']
    panel_data = []
    
    for item in items:
        # Base trend with some randomness
        trend = np.arange(500) * 0.01 + np.random.normal(0, 0.5, 500).cumsum() * 0.1
        # Weekly seasonality
        seasonality = 5 * np.sin(np.arange(500) * 2*np.pi/7)
        # Combine with noise
        sales = trend + seasonality + np.random.normal(0, 1, 500)
        # Ensure non-negative
        sales = np.maximum(sales, 0) + 10
        
        df = pd.DataFrame({
            'date': dates,
            'item_id': item,
            'sold': sales
        })
        panel_data.append(df)
        
    panel_df = pd.concat(panel_data, ignore_index=True)
    
panel_df['date'] = pd.to_datetime(panel_df['date'])

def remove_tail(df, drop_value=28):
    """Remove the last n observations from each group in a panel dataset"""
    assert 'item_id' in df.columns and 'date' in df.columns
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-drop_value]
    df_filter = df_sorted.groupby('item_id').apply(remove_tail_values).reset_index(drop=True)
    return df_filter

# Remove the last 28 days (4 weeks) from each product time series
panel_df = remove_tail(panel_df, drop_value=28).copy()

# %%
#| label: Plot single series
#| echo: true
#| eval: true
#| fig-cap: 'Product sales of FOODS_2_197 over time'
#| fig-cap-location: bottom

# Extract a single product series for analysis
series_df = panel_df.loc[panel_df.item_id == 'FOODS_2_197'].copy()
series_df['date'] = pd.to_datetime(series_df['date'])

# Plot the time series
fig, ax = darts_plot(series_df, 'date', ['sold'], 
                    labels=['Sales'],
                    title='Sales for FOODS_2_197')
plt.show()

# %%
#| label: Decomposition plot
#| echo: true
#| eval: true
#| fig-cap: 'Decomposition of FOODS_2_197 time series'
#| fig-cap-location: bottom

# Set date as index for time series decomposition
series_df.set_index('date', inplace=True, drop=True)

# Focus on a shorter period for clearer decomposition
short_series = series_df['2015':'2016'].copy()

# Plot decomposition
fig = plot_seasonal_decompose(short_series['sold'], model='additive')
plt.show()

# %%
#| label: Stationarity checks
#| echo: true
#| eval: true

# Augmented Dickey-Fuller test for stationarity
result = adfuller(series_df['sold'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# KPSS test for stationarity (alternative test)
kpss_stat, p_value, lags, crit = kpss(series_df['sold'])
print('\nKPSS Statistic: %f' % kpss_stat)
print('p-value: %f' % p_value)
print('Critical Values:')
for key, value in crit.items():
    print('\t%s: %.3f' % (key, value))

# %%
#| label: Plotting autocorrelation
#| echo: true
#| eval: true
#| fig-cap: 'Autocorrelations for FOODS_2_197, with white noise threshold'
#| fig-cap-location: bottom

# Create side-by-side ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ACF plot
plot_acf(series_df['sold'], lags=32, ax=ax1, color=COLORS["blue"])
ax1.set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(series_df['sold'], lags=32, ax=ax2, color=COLORS["blue"])
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# %%
#| label: Plotting residuals
#| echo: false
#| eval: true
#| fig-cap: 'Residual plot for FOODS_2_197'
#| fig-cap-location: bottom

# Reset index to access date column
series_df.reset_index(inplace=True)

# Convert date to numeric (days since the start)
series_df['days'] = (pd.to_datetime(series_df['date']) - pd.to_datetime(series_df['date'].min())).dt.days

# Fit linear regression model for trend
model = LinearRegression()
X = series_df[['days']]
y = series_df['sold']
model.fit(X, y)

# Create predictions and calculate residuals
series_df['predicted'] = model.predict(X)
series_df['residuals'] = series_df['sold'] - series_df['predicted']

# Plot residuals
plt.figure(figsize=(12, 6))
plt.scatter(series_df['date'], series_df['residuals'], alpha=0.5, color=COLORS["black"])
plt.axhline(y=0, color=COLORS["blue"], linestyle='-', linewidth=2)
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
#| label: Simple split
#| echo: true
#| eval: true

# Create a simple train-test split
from dateutil.relativedelta import relativedelta
last_date = series_df['date'].max()
split_date = last_date - relativedelta(months=3)

# Split the data
train_df = series_df[series_df['date'] < split_date]
test_df = series_df[series_df['date'] >= split_date]

# %%
#| label: Simple split plotting
#| echo: false
#| eval: true
#| fig-cap: 'Simple train-test split'
#| fig-cap-location: bottom

# Plot the train-test split
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color=COLORS["black"])
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color=COLORS["blue"])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
#| label: Train-validation-test
#| echo: true
#| eval: true

# Find split points for train-validation-test
last_date = series_df['date'].max()
test_split_date = last_date - relativedelta(months=3)
val_split_date = test_split_date - relativedelta(months=3)

# Create train, validation, and test sets
train_df = series_df[series_df['date'] < val_split_date]
val_df = series_df[(series_df['date'] >= val_split_date) & (series_df['date'] < test_split_date)]
test_df = series_df[series_df['date'] >= test_split_date]

# %%
#| label: Train-validation-test plotting
#| echo: true
#| eval: true
#| fig-cap: 'Train-validation-test split'
#| fig-cap-location: bottom

# Plot the train-validation-test split
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color=COLORS["black"])
plt.plot(val_df['date'], val_df['sold'], label='Validation Data', color=COLORS["purple"])
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color=COLORS["blue"])
plt.tight_layout()
plt.title('Train-Validation-Test Split')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# %%
#| label: Expanding windows CV plot
#| echo: false
#| eval: true
#| fig-cap: 'Expanding Window Time Series Cross-Validation'
#| fig-cap-location: bottom

# Set up the plot
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Get last date and calculate split points
last_date = series_df['date'].max()
test_periods = [relativedelta(months=3 * i) for i in range(3, 0, -1)]  # Reversed order

# Create dummy lines for the legend
train_line = mlines.Line2D([], [], color=COLOR_LIST[0], label='Training Data')
test_line = mlines.Line2D([], [], color=COLOR_LIST[1], label='Test Data')
excluded_line = mlines.Line2D([], [], color='gray', alpha=0.5, label='Excluded Data')
legend_items = [train_line, test_line, excluded_line]

for fold, (ax, test_period) in enumerate(zip(reversed(axs), test_periods), 1):  # Reversed axs
    # Calculate split dates
    test_end = last_date
    test_start = test_end - relativedelta(months=3)
    train_end = test_start
    
    # Create split point
    train_df = series_df[series_df['date'] < train_end]
    test_df = series_df[(series_df['date'] >= test_start) & (series_df['date'] < test_end)]
    excluded_df = series_df[series_df['date'] >= test_end]

    # Plot training data
    ax.plot(train_df['date'], train_df['sold'], color=COLOR_LIST[0])
    
    # Plot test data
    ax.plot(test_df['date'], test_df['sold'], color=COLOR_LIST[1])
    
    # Plot excluded data (previously used test data)
    if not excluded_df.empty:
        ax.plot(excluded_df['date'], excluded_df['sold'], color='gray', alpha=0.5)

    # highlight test period
    ax.axvspan(test_start, test_end, alpha=0.2, color=COLOR_LIST[1])

    # Labels
    ax.set_ylabel('Sales')
    ax.set_title(f'Fold {4-fold}')  
    ax.grid(True)

    # Update last_date for next iteration
    last_date = test_start

# Create common x-label
fig.text(0.5, 0.01, 'Date', ha='center', va='center')

# Add the legend to the figure
fig.legend(handles=legend_items, loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), ncol=3)

# Adjust layout
plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])  
plt.show()

# %%
#| label: Rolling windows CV plot
#| echo: false
#| eval: true
#| fig-cap: 'Rolling Window Time Series Cross-Validation'
#| fig-cap-location: bottom

# Set up the plot
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Create dummy lines for the legend
train_line = mlines.Line2D([], [], color=COLOR_LIST[0], label='Training Data')
test_line = mlines.Line2D([], [], color=COLOR_LIST[1], label='Test Data')
excluded_line = mlines.Line2D([], [], color='gray', alpha=0.5, label='Excluded Data')
legend_items = [train_line, test_line, excluded_line]

# Get last date and calculate split points
last_date = series_df['date'].max()
test_periods = [relativedelta(months=3 * i) for i in range(3, 0, -1)]  

for fold, (ax, test_period) in enumerate(zip(reversed(axs), test_periods), 1): 
    # Calculate split dates
    test_end = last_date
    test_start = test_end - relativedelta(months=3)
    train_end = test_start
    train_start = train_end - relativedelta(years=2)  # 2-year rolling window
    
    # Create split point
    train_df = series_df[(series_df['date'] >= train_start) & (series_df['date'] < train_end)]
    test_df = series_df[(series_df['date'] >= test_start) & (series_df['date'] < test_end)]
    excluded_before_df = series_df[series_df['date'] < train_start]
    excluded_after_df = series_df[series_df['date'] >= test_end]

    # Plot excluded data before training period
    if not excluded_before_df.empty:
        ax.plot(excluded_before_df['date'], excluded_before_df['sold'], color='gray', alpha=0.5)

    # Plot training and test data
    ax.plot(train_df['date'], train_df['sold'], color=COLOR_LIST[0])
    ax.plot(test_df['date'], test_df['sold'], color=COLOR_LIST[1])
    
    # Plot excluded data after test
    if not excluded_after_df.empty:
        ax.plot(excluded_after_df['date'], excluded_after_df['sold'], color='gray', alpha=0.5)

    # Highlight training and test periods
    ax.axvspan(train_start, train_end, alpha=0.1, color=COLOR_LIST[0])
    ax.axvspan(test_start, test_end, alpha=0.2, color=COLOR_LIST[1])

    # Labels
    ax.set_ylabel('Sales', fontsize=20)
    ax.set_title(f'Fold {4-fold}', fontsize=20)
    ax.grid(True)

    # Update last_date for next iteration
    last_date = test_start

# Common x-axis label
fig.text(0.53, 0.01, 'Date', ha='center', va='center', fontsize=20)

# Add the shared legend to the figure
fig.legend(handles=legend_items, loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=18)

# Adjust layout
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()

# %%
#| label: Error metrics visualization
#| echo: false
#| eval: true
#| fig-cap: 'Absolute error weighting plot'
#| fig-cap-location: bottom

# Visualize how absolute error weights different types of errors
fig, ax = plt.subplots(figsize=(12, 6))

# Error range
error = np.linspace(-100, 100, 1000)
absolute_error = np.abs(error)

# Plotting
ax.plot(error, absolute_error, color=COLORS["black"], linewidth=4)
ax.fill_between(error, 0, absolute_error, color="#ff6933", alpha=0.95)

# Labels
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Absolute Error', fontsize=18)

ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# %%
#| label: Squared Error (SE) weight plot
#| echo: false
#| eval: true
#| fig-cap: 'Squared Error (SE) weight plot'
#| fig-cap-location: bottom

# Visualize how squared error weights different types of errors
fig, ax = plt.subplots(figsize=(12, 6))

# Error range
error = np.linspace(-10, 10, 1000)
squared_error = error ** 2

# Plotting
ax.plot(error, squared_error, color=COLORS["black"], linewidth=4)
ax.fill_between(error, 0, squared_error, color="#ff6933", alpha=0.95)

# Labels
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Squared Error', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Limit y-axis 
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# %%
#| label: AE and SE comparison
#| echo: false
#| eval: true
#| fig-cap: 'Comparison of Absolute Error and Squared Error weightings'
#| fig-cap-location: bottom

# Compare absolute error and squared error
actual = 5
forecasts = np.linspace(0, 10, 100)

# Calculate errors
absolute_error = np.abs(forecasts - actual)
squared_error = (forecasts - actual)**2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecasts, absolute_error, label='Absolute Error', color=COLORS["blue"], linewidth=3)
ax.plot(forecasts, squared_error, label='Squared Error', color=COLORS["purple"], linewidth=3)

ax.set_xlabel('Forecast Value', fontsize=14)
ax.set_ylabel('Error', fontsize=14)
ax.set_title('Comparison of Absolute Error and Squared Error', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %%
#| label: Scale invariance
#| echo: true
#| eval: true

# Demonstrate scale invariance of error metrics
# Define data
actual = np.array([100, 150, 200, 250, 300])
forecast = np.array([105, 160, 190, 240, 310])

# Define error function
def mse(actual, forecast):
    return np.mean((actual - forecast)**2)

# Calculate MSE
mse_original = mse(actual, forecast)
mse_scaled = mse(actual * 10, forecast * 10)

print(f"MSE (original): {mse_original:.2f}")
print(f"MSE (scaled): {mse_scaled:.2f}")
print(f"Ratio of scaled to original MSE: {mse_scaled / mse_original:.2f}")

# %%
#| label: Plot and accuracy of naive models
#| echo: false
#| eval: true
#| fig-cap: '3-Month Validation: Naive vs Seasonal Naive Forecast'
#| fig-cap-location: bottom

# Prepare data for naive model evaluation
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

# Set up plot
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(15, 8))

# Validation period
period_length = pd.Timedelta(days=90)

def naive_forecast(train_data, periods):
    """Simple naive forecast using the last observed value"""
    return [train_data.iloc[-1]] * periods

def seasonal_naive_forecast(train_data, periods, season_length):
    """Seasonal naive forecast using values from the previous season"""
    forecast = []
    for i in range(periods):
        forecast.append(train_data.iloc[-(season_length - i % season_length)])
    return forecast

# Calculate split dates
test_end = df['date'].max()
test_start = test_end - period_length
train_end = test_start

# Create split point
train_df = df[df['date'] < train_end]
test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]

# Generate forecasts
naive_forecast_values = naive_forecast(train_df['sold'], len(test_df))
seasonal_naive_forecast_values = seasonal_naive_forecast(train_df['sold'], len(test_df), 14)

# Calculate metrics
naive_metrics = calculate_metrics(test_df['sold'], naive_forecast_values)
seasonal_naive_metrics = calculate_metrics(test_df['sold'], seasonal_naive_forecast_values)

# Plot data and forecasts
ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=COLOR_LIST[0])
ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=COLOR_LIST[1])
ax.plot(test_df['date'], naive_forecast_values, label='Naive Forecast', color=COLOR_LIST[2], linestyle='--')
ax.plot(test_df['date'], seasonal_naive_forecast_values, label='Seasonal Naive Forecast', color=COLOR_LIST[4], linestyle='--')

# Highlight test period
ax.axvspan(test_start, test_end, alpha=0.2, color=COLOR_LIST[1])

# Labels
ax.set_ylabel('Sales')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
ax.grid(True)

# Add error metrics to plot
metric_names = ['MAE', 'RMSE', 'MAPE']
naive_text = '\n'.join([f'Naive {name}: {value:.2f}' for name, value in zip(metric_names, naive_metrics)])
seasonal_text = '\n'.join([f'Seasonal Naive {name}: {value:.2f}' for name, value in zip(metric_names, seasonal_naive_metrics)])

ax.text(0.02, 0.98, naive_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=COLOR_LIST[2])
ax.text(0.02, 0.84, seasonal_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=COLOR_LIST[4])

plt.tight_layout()
plt.show()

# %%
#| label: Comparing forecast lengths plot
#| echo: false
#| eval: true
#| fig-cap: 'Effect of validation length on forecast accuracy'
#| fig-cap-location: bottom

# Select data for the experiment
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

# Set up plot
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle('Effect of Validation Period Length on Forecast Accuracy', fontsize=20)

# Validation periods
validation_periods = [
    ('2 Weeks', 14),       # 14 days
    ('3 Months', 90),      # 90 days
    ('6 Months', 180)      # 180 days
]

# Compare different validation period lengths
for i, (ax, (period_name, days)) in enumerate(zip(axs, validation_periods)):
    # Calculate split dates
    test_end_date = df['date'].max()
    test_start_date = test_end_date - pd.DateOffset(days=days)
    
    # Create split point
    train_df = df[df['date'] < test_start_date].copy()
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
    
    # Fit exponential smoothing model
    train_ts = train_df.set_index('date')['sold']
    
    model = ExponentialSmoothing(
        train_ts, 
        seasonal_periods=7,  # Weekly seasonality
        trend='add', 
        seasonal='add'
    ).fit()
    
    # Generate forecast
    forecast = model.forecast(len(test_df))
    
    # Align forecast with test dates
    forecast = pd.Series(
        forecast.values, 
        index=test_df['date']
    )
    
    # Calculate errors
    mape = mean_absolute_percentage_error(test_df['sold']+1, forecast.values+1)
    mae = mean_absolute_error(test_df['sold'], forecast.values)
    
    # Plot data
    ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
    ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=COLOR_LIST[0])
    ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=COLOR_LIST[1])
    ax.plot(test_df['date'], forecast.values, 
            label=f'Exp. Smoothing (MAE: {mae:.2f}, MAPE: {mape:.2%})', 
            color=COLOR_LIST[2], linestyle='--')
    
    # Highlight test period
    ax.axvspan(test_start_date, test_end_date, alpha=0.2, color=COLOR_LIST[1])
    
    # Labels
    ax.set_ylabel('Sales')
    ax.set_title(f'{period_name} Validation Period')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

# Set x-label
fig.text(0.5, 0.04, 'Date', ha='center', va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.show()

# %%
#| label: Data leakage visualization
#| echo: false
#| eval: true
#| fig-cap: 'Impact of splitting before and after data transformation'
#| fig-cap-location: bottom

# Generate a dataset with a trend change
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
split_idx = 270  # '2020-09-27'
split_date = dates[split_idx]

# Create training vs. test indices
train_indices = np.arange(split_idx)
test_indices = np.arange(split_idx, 365)

# Training period trend: moderate upward quadratic
train_trend = 10 + 0.05 * train_indices + 0.0002 * train_indices**2

# Test period trend: abrupt downward
test_trend_start = train_trend[-1] 
test_trend = test_trend_start - 0.1*(test_indices - test_indices[0]) \
                             - 0.001*(test_indices - test_indices[0])**2

# Combine
trend = np.concatenate([train_trend, test_trend])

# Seasonal + noise
seasonal = 5 * np.sin(np.arange(365) * 2*np.pi/7)  # weekly seasonality
noise = np.random.normal(0, 1, 365)
values = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'value': values})
train_df = df[df['date'] < split_date].copy()
test_df  = df[df['date'] >= split_date].copy()

# Wrong detrending (data leakage)
from scipy import signal
df_wrong = df.copy()
detrended_full = signal.detrend(df_wrong['value'])
df_wrong['detrended'] = detrended_full

train_wrong = df_wrong[df_wrong['date'] < split_date].copy()
test_wrong  = df_wrong[df_wrong['date'] >= split_date].copy()

# Fit model on wrongly-detrended data
from statsmodels.tsa.arima.model import ARIMA
model_wrong = ARIMA(train_wrong['detrended'], order=(7,1,7)).fit()
# Forecast in detrended space
fcast_wrong_detrended = model_wrong.forecast(len(test_wrong))

# Add back the full-data linear trend
mean_diff_full = df_wrong['value'].mean() - df_wrong['detrended'].mean()
fcast_wrong = fcast_wrong_detrended + mean_diff_full

# Correct detrending (no leakage)
train_right = train_df.copy()
train_right['time_idx'] = np.arange(len(train_right))
X_train = train_right[['time_idx']]
y_train = train_right['value']

lr = LinearRegression().fit(X_train, y_train)
train_trend_pred = lr.predict(X_train)
train_right['detrended'] = train_right['value'] - train_trend_pred

# For test data, use training trend model
test_right = test_df.copy()
test_right['time_idx'] = np.arange(len(train_right), len(train_right)+len(test_right))
test_trend_pred = lr.predict(test_right[['time_idx']])
test_right['detrended'] = test_right['value'] - test_trend_pred

# Fit model on correctly detrended data
model_right = ARIMA(train_right['detrended'], order=(7,1,7)).fit()
fcast_right_detrended = model_right.forecast(len(test_right))

# Add back the training-derived trend
fcast_right = fcast_right_detrended + test_trend_pred

# Calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_wrong = calculate_rmse(test_df['value'], fcast_wrong)
rmse_right = calculate_rmse(test_df['value'], fcast_right)

# Create plot
plt.figure(figsize=(12, 6))

plt.plot(df['date'], df['value'], color=COLOR_LIST[0], label='Original time series')
plt.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')
plt.plot(test_df['date'], fcast_wrong, color=COLOR_LIST[1], label='Forecast (Leaked Trend)', linestyle='--')
plt.plot(test_df['date'], fcast_right, color=COLOR_LIST[2], label='Forecast (No Leakage)', linestyle=':')

# Highlight training and test periods
plt.axvspan(df['date'].min(), split_date, alpha=0.1, color=COLOR_LIST[0])  # Training period
plt.axvspan(split_date, df['date'].max(), alpha=0.2, color=COLOR_LIST[1])  # Test period

# Title and labels
plt.title(f'RMSE: Leaked Trend = {rmse_wrong:.2f}, No Leakage = {rmse_right:.2f}', fontsize=18)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Values', fontsize=20)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# %%
#| label: Target feature leakage visualization
#| echo: false
#| eval: true
#| fig-cap: 'Impact of lookforward bias via lagged features'
#| fig-cap-location: bottom

# Generate sample data for lagged feature experiment
np.random.seed(123)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100))  # Random walk
df = pd.DataFrame({'date': dates, 'value': values})

# Add lag features
df['lag1'] = df['value'].shift(1)
df['lag2'] = df['value'].shift(2)
df = df.dropna()

# Split data
split_idx = 80
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Train a simple model
model = LinearRegression()
X_train = train_df[['lag1', 'lag2']]
y_train = train_df['value']
model.fit(X_train, y_train)

# WRONG WAY: Using actual values for recursive forecasting
forecasts_wrong = []
for i in range(len(test_df)):
    # Get actual lags from test set (leakage!)
    X_test = test_df.iloc[i:i+1][['lag1', 'lag2']] 
    pred = model.predict(X_test)[0]
    forecasts_wrong.append(pred)

# RIGHT WAY: Recursive forecasting without leakage
forecasts_right = []
# Initialize with last values from training set
last_values = train_df['value'].tail(2).values
for i in range(len(test_df)):
    # Create features from previous predictions
    X_test = np.array([[last_values[-1], last_values[-2]]])
    pred = model.predict(X_test)[0]
    forecasts_right.append(pred)
    # Update for next step forecast
    last_values = np.append(last_values[1:], pred)

# Plot comparisons
plt.figure(figsize=(12, 8))

# First subplot - Wrong approach
plt.subplot(2, 1, 1)
plt.plot(train_df['date'], train_df['value'], color=COLOR_LIST[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=COLOR_LIST[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_wrong, color=COLOR_LIST[1], linestyle='--', label='Forecasts with Leakage')

# Highlight test period
plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=COLOR_LIST[1])
plt.axvline(x=test_df['date'].min(), color='r', linestyle='--')

plt.title('WRONG: Using Actual Values for Lagged Features (Leakage!)', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Second subplot - Correct approach
plt.subplot(2, 1, 2)
plt.plot(train_df['date'], train_df['value'], color=COLOR_LIST[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=COLOR_LIST[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_right, color=COLOR_LIST[2], linestyle=':', label='Forecasts without Leakage')

# Highlight test period
plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=COLOR_LIST[2])
plt.axvline(x=test_df['date'].min(), color='r', linestyle='--')

plt.title('CORRECT: Recursive Forecasting Using Only Predictions', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Calculate RMSE for both approaches
rmse_wrong = np.sqrt(mean_squared_error(test_df['value'], forecasts_wrong))
rmse_right = np.sqrt(mean_squared_error(test_df['value'], forecasts_right))

# Add a text annotation showing the RMSE
plt.figtext(0.5, 0.01, f'RMSE: Leaked Features = {rmse_wrong:.2f}, No Leakage = {rmse_right:.2f}', 
            ha='center', fontsize=14, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)  # Make room for the RMSE text
plt.show()