# Extracted from chapter3_Evaluation_ts.qmd
# Do not edit the source .qmd file directly.

#| label: Libraries and plotting function
#| message: false
#| echo: false
#| eval: true

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from dateutil.relativedelta import relativedelta
from cycler import cycler

from pathlib import Path 
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18

# general settings
class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18
    
    
# adjust the parameters for displayed figures    
plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)}) 

custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]

def darts_plot(df, x_column, y_columns, labels=None, quantiles=None, title=None, fontsize=18):
    # Set the default font size for all text elements
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if labels is None:
        labels = y_columns
    
    alpha_confidence_intvls = 0.25

    # Define the color palette to match darts
    darts_colors = custom_palette

    for i, y_col in enumerate(y_columns):
        color = darts_colors[i % len(darts_colors)]
        
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

# ----------------------------------------------------------------------

#| label: Bias-variance tradeoff
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.1: Bias-variance tradeoff" 
#| fig-cap-location: bottom

# Generate data points
x = np.linspace(0, 10, 100)

# Define functions for bias, variance, and total error
bias = 10 * np.exp(-0.5 * x)
variance = 0.05 * (x ** 2)
total_error = bias + variance

# Plotting
plt.figure(figsize=(10, 6))

# Plot curves
# colours = "black", "003DFD", "b512b8", "11a9ba", "0d780f", "f77f07", "ba0f0f"
plt.plot(x, bias, color='black', label='Bias²')
plt.plot(x, variance, color='#003DFD', label='Variance', linestyle='dashdot')
plt.plot(x, total_error, color='#b512b8', label='Total Error', linestyle=':')

# Vertical line for optimal complexity
optimal_x = (x[np.argmin(total_error)]) -0.2
plt.axvline(x=optimal_x, color='#11a9ba', linestyle='--', label='Optimal Model Complexity')

# Labels etc
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend(fontsize=12)
plt.text(0.5, 9, 'Bias²', color='black', fontsize=12)
plt.text(8, 2.8, 'Variance', color='#003DFD', fontsize=12)
plt.text(5, 3, 'Total Error', color='#b512b8', fontsize=12)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# ----------------------------------------------------------------------

#| label: Overfitting plot 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.2: Training and validation loss"
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
plt.plot(iterations, training_loss, color="black", label='Training loss', linewidth=2)
plt.plot(iterations, validation_loss, color="#003DFD", label='Validation loss', linewidth=2)

# Vertical line at start of overfitting
plt.axvline(x=overfitting_point, color="#11a9ba", linestyle='--', linewidth=2)

# Annotation
plt.text(overfitting_point + 2, 1.2, 'Overfitting starts from here', 
         rotation=0, verticalalignment='center', fontsize=12, color='#11a9ba')

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

# ----------------------------------------------------------------------

#| label: Data wrangling
#| message: false
#| echo: true
#| eval: true
# Data wrangling
panel_df = pd.read_csv(CFG.data_folder / 'M5_t20_ABC.csv', index_col=False).drop(columns = 'Unnamed: 0')
panel_df['date'] = pd.to_datetime(panel_df['date'])
panel_df.set_index('date')

def remove_tail(df, drop_value=28):
    assert 'item_id' in df.columns and 'date' in df.columns
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-drop_value]
    df_filter = df_sorted.groupby('item_id').apply(remove_tail_values).reset_index(drop=True)
    return df_filter

panel_df = remove_tail(panel_df, drop_value=28).copy()

# ----------------------------------------------------------------------

#| label: Plot single series
#| message: false
#| echo: true
#| eval: true
#| fig-cap: "3.3: Product sales of FOODS_2_197 over time"
#| fig-cap-location: bottom

# Single product series

series_df = panel_df.loc[panel_df.item_id == 'FOODS_2_197'].copy()
series_df['date'] = pd.to_datetime(series_df['date'])

fig, ax = darts_plot(series_df, 'date', ['sold'], 
                             labels=['Sales'],
                             title='Sales for FOODS_2_197')
plt.show()

# ----------------------------------------------------------------------

#| label: Decompostion plotting function
#| message: false
#| echo: false
#| eval: true

def plot_seasonal_decompose(series, model='additive', figsize=(12, 10), fontsize=18):
    # Set global font size
    plt.rcParams.update({'font.size': fontsize})
    
    decomposition = seasonal_decompose(series, model=model)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Observed component
    decomposition.observed.plot(ax=ax1, color='#000000', linewidth=0.5)
    ax1.set_ylabel('Observed', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)  # Slightly smaller for tick labels
    ax1.set_title('Observed Time Series', fontsize=fontsize)
    
    # Trend component
    decomposition.trend.plot(ax=ax2, color='#003DFD', linewidth=1)
    ax2.set_ylabel('Trend', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.set_title('Trend Component', fontsize=fontsize)
    
    # Seasonal component
    decomposition.seasonal.plot(ax=ax3, color='#b512b8', linewidth=0.5)
    ax3.set_ylabel('Seasonal', fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax3.set_title('Seasonal Component', fontsize=fontsize)
    
    # Residual component
    ax4.scatter(decomposition.resid.index, decomposition.resid, color='#11a9ba', s=3, alpha=0.5)
    ax4.set_ylabel('Residual', fontsize=fontsize)
    ax4.set_xlabel('Date', fontsize=fontsize)  # Added x-label for bottom subplot
    ax4.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax4.set_title('Residual Component', fontsize=fontsize)
    
    # Improve layout with more space for the larger text
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)  # Add more space between subplots
    
    return fig

# ----------------------------------------------------------------------

#| label: Decompostion plot
#| message: false
#| echo: true
#| eval: true
#| fig-cap: "3.4: Decompostion of FOODS_2_197 time series"
#| fig-cap-location: bottom

series_df.set_index('date', inplace=True, drop=True)
short_series = series_df['2015':'2016'].copy()
fig = plot_seasonal_decompose(short_series['sold'], model='additive')
plt.show()

# ----------------------------------------------------------------------

#| label: StationaryCheck 1 
#| message: false
#| echo: true
#| eval: true
from statsmodels.tsa.stattools import adfuller

result = adfuller(series_df['sold'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ----------------------------------------------------------------------

#| label: StationaryCheck 2
#| message: false
#| echo: false
#| eval: true
from statsmodels.tsa.stattools import kpss

kpss_stat, p_value, lags, crit = kpss(series_df['sold'])
print(f'KPSS Statistic: {kpss_stat}')
print(f'p-value: {p_value}')
print('Critical Values:')
for key, value in crit.items():
    print(f'\t{key}: {value}')

# ----------------------------------------------------------------------

#| label: Plotting autocorrelation
#| message: false
#| echo: true
#| eval: true
#| fig-cap: "3.5: Autocorrelations for FOODS_2_197, with whitenoise threshold" 
#| fig-cap-location: bottom
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Facet plotting 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ACF plot
plot_acf(series_df['sold'], lags=32, ax=ax1, color= '#003DFD')
ax1.set_title('Autocorrelation Function (ACF)')
# PACF plot
plot_pacf(series_df['sold'], lags=32, ax=ax2, color= '#003DFD')
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Plotting residuals
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.6: Residual plot for FOODS_2_197"
#| fig-cap-location: bottom

from sklearn.linear_model import LinearRegression
series_df.reset_index(inplace=True)
# Convert date to numeric (days since the start)
series_df['days'] = (pd.to_datetime(series_df['date']) - pd.to_datetime(series_df['date'].min())).dt.days

# Fit linear regression model
model = LinearRegression()
X = series_df[['days']]
y = series_df['sold']
model.fit(X, y)
# Create predictions
series_df['predicted'] = model.predict(X)
# Calculate residuals
series_df['residuals'] = series_df['sold'] - series_df['predicted']

# Plot residuals
plt.figure(figsize=(12, 6))
plt.scatter(series_df['date'], series_df['residuals'], alpha=0.5, color = '#000000')
plt.axhline(y=0, color='#003DFD', linestyle='-', linewidth = 2)
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Simple split
#| message: false
#| echo: true
#| eval: true

from dateutil.relativedelta import relativedelta
last_date = series_df['date'].max()
split_date = last_date - relativedelta(months=3)

# Create split point
train_df = series_df[series_df['date'] < split_date]
test_df = series_df[series_df['date'] >= split_date]

# ----------------------------------------------------------------------

#| label: Simple split plotting
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.7: Simple train-test split"
#| fig-cap-location: bottom
# Plot
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color='black')
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color='#003DFD')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Train-validation-test 
#| message: false
#| echo: true
#| eval: true

# Find split points
last_date = series_df['date'].max()
test_split_date = last_date - relativedelta(months=3)
val_split_date = test_split_date - relativedelta(months=3)

# Create train, validation, and test sets
train_df = series_df[series_df['date'] < val_split_date]
val_df = series_df[(series_df['date'] >= val_split_date) & (series_df['date'] < test_split_date)]
test_df = series_df[series_df['date'] >= test_split_date]

# ----------------------------------------------------------------------

#| label: Train-validation-test plotting
#| message: false
#| echo: true
#| eval: true
#| fig-cap: "3.8: Train-validation-test split"
#| fig-cap-location: bottom

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color='black')
plt.plot(val_df['date'], val_df['sold'], label='Validation Data', color='#b512b8')
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color='#003DFD')
plt.tight_layout()
plt.title('Train-Validation-Test Split')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------

#| label: Expanding windows CV plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.9: Expanding Window Time Series Cross-Validation"
#| fig-cap-location: bottom

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)  # Increased height

# Palette
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Get last date and calculate split points
last_date = series_df['date'].max()
test_periods = [relativedelta(months=3 * i) for i in range(3, 0, -1)]  # Reversed order

# Create dummy lines for the legend
train_line = mlines.Line2D([], [], color=colors[0], label='Training Data')
test_line = mlines.Line2D([], [], color=colors[1], label='Test Data')
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
    ax.plot(train_df['date'], train_df['sold'], color=colors[0])
    
    # Plot test data
    ax.plot(test_df['date'], test_df['sold'], color=colors[1])
    
    # Plot excluded data (previously used test data)
    if not excluded_df.empty:
        ax.plot(excluded_df['date'], excluded_df['sold'], color='gray', alpha=0.5)

    # highlight test period
    ax.axvspan(test_start, test_end, alpha=0.2, color=colors[1])

    # Labels etc
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

# ----------------------------------------------------------------------

#| label: Rolling windows CV plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.10: Rolling Window Time Series Cross-Validation"
#| fig-cap-location: bottom

# Set up the plot
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Create dummy lines for the legend
train_line = mlines.Line2D([], [], color=colors[0], label='Training Data')
test_line = mlines.Line2D([], [], color=colors[1], label='Test Data')
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
    ax.plot(train_df['date'], train_df['sold'], color=colors[0])
    ax.plot(test_df['date'], test_df['sold'], color=colors[1])
    
    # Plot excluded data after test
    if not excluded_after_df.empty:
        ax.plot(excluded_after_df['date'], excluded_after_df['sold'], color='gray', alpha=0.5)

    # Highlight training and test periods
    ax.axvspan(train_start, train_end, alpha=0.1, color=colors[0])
    ax.axvspan(test_start, test_end, alpha=0.2, color=colors[1])

    # Labels etc
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
plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Adjust the rect parameter to make room for title and legend
plt.show()

# ----------------------------------------------------------------------

#| label: Absolute error weighting plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.11: Absolute error weighting plot"
#| fig-cap-location: bottom

fig, ax = plt.subplots(figsize=(12, 6))

# Colours
colors = ["black", "#ff6933"]

# Generate data
error = np.linspace(-100, 100, 1000)
absolute_error = np.abs(error)

# Plotting
ax.plot(error, absolute_error, color=colors[0], linewidth=4)
ax.fill_between(error, 0, absolute_error, color=colors[1], alpha=0.95)

# Lables etc
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Absolute Error', fontsize=18)

ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: model comparison plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.12: Comparison of True Values and Predictions (A vs B) with MAPE"
#| fig-cap-location: bottom
t = np.linspace(0, 2, 3)

# Define the true, A, and B values based on given equations
yt_actuals = 0.01 + 0.1 * t
yt_A = 0.02 + 0.1 * t
yt_B = 0.01005 + 0.135 * t + 0.005 * t**2

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(t, yt_actuals, label='Actuals', color='black', linewidth=3)
plt.plot(t, yt_A, label='A: Good model', color='#003DFD', linestyle=':', linewidth=3)
plt.plot(t, yt_B, label='B: Poor model', color='#b512b8', linestyle='--', linewidth=3)

# Adding labels and title
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Y value', fontsize=18)

ax.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: model comparison with MAPE
#| message: false
#| echo: true
#| eval: true
def mape(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

mape_A = mape(yt_actuals, yt_A)
mape_B = mape(yt_actuals, yt_B)

print(f'Model A has a MAPE of {mape_A}') 
print(f'Model B has a MAPE score of {mape_B}')

# ----------------------------------------------------------------------

#| label: model comparison with shifted MAPE
#| message: false
#| echo: true
#| eval: true

mape_A = mape(yt_actuals + 1, yt_A + 1)
mape_B = mape(yt_actuals + 1, yt_B + 1)

print(f'Model A has a shifted MAPE of {mape_A}') 
print(f'Model B have a shifted MAPE score of {mape_B}')

# ----------------------------------------------------------------------

#| label: APE error weighting plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.13: APE error weighting plot. Adapted from @manuMODERNTIMESERIES2022"
#| fig-cap-location: bottom
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Absolute Error', fontsize=20)

# Colours
colors = ["black", "#ff6933"]

actuals = np.linspace(1, 10, 100)
forecasts = np.linspace(10, 1, 100)
percent_error = np.abs((actuals - forecasts) / actuals) * 100

# Plotting
ax.plot(actuals, percent_error, color=colors[0], linewidth=4)
ax.fill_between(actuals, 0, percent_error, color=colors[1], alpha=0.95)

# Labels etc
ax.set_xlabel('Actuals', fontsize=18)
ax.set_ylabel('Absolute Percent Error', fontsize=18)

ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set x-axis ticks and labels
x_ticks = np.arange(1, 11, 1)
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'A:{a}\nF:{f}' for a, f in zip(x_ticks, x_ticks[::-1])], fontsize=18)

# Set y-axis limit
ax.set_ylim(0, 900)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: APE unit change comparison 
#| message: false
#| echo: false
#| eval: true
# Define data
actual_celsius = np.array([20, 25, 30, 35, 40])
forecast_celsius = np.array([22, 24, 31, 33, 41])

# Convert to kelvin
actual_kelvin = actual_celsius + 273.15
forecast_kelvin = forecast_celsius + 273.15

# Define error functions
def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def wape(actual, forecast):
    return np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual)) * 100

def mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))

# Calculate errors
mape_celsius = mape(actual_celsius, forecast_celsius)
mape_kelvin = mape(actual_kelvin, forecast_kelvin)
wape_celsius = wape(actual_celsius, forecast_celsius)
wape_kelvin = wape(actual_kelvin, forecast_kelvin)
mae_celsius = mae(actual_celsius, forecast_celsius)
mae_kelvin = mae(actual_kelvin, forecast_kelvin)

# Print results
metrics = {
    'Metric': ['MAPE', 'WAPE', 'MAE'],
    'Celsius': [mape_celsius, wape_celsius, mae_celsius],
    'Kelvin': [mape_kelvin, wape_kelvin, mae_kelvin]
}
metrics_df = pd.DataFrame(metrics)
metrics_df['Celsius'] = metrics_df['Celsius'].apply(lambda x: f"{x:.2f}{'%' if x != mae_celsius else ''}")
metrics_df['Kelvin'] = metrics_df['Kelvin'].apply(lambda x: f"{x:.2f}{'%' if x != mae_kelvin else ''}")

# Display table
# Create DataFrame and display as HTML table
metrics_df = pd.DataFrame(metrics)
# Format the values
for col in ['Celsius', 'Kelvin']:
    metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}{'%' if x != mae_celsius and x != mae_kelvin else ''}")

# Display as HTML table
from IPython.display import display, HTML
display(HTML(metrics_df.to_html(index=False, classes="table table-striped")))

# ----------------------------------------------------------------------

#| label: APE unit change comparison plotted
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.14: APE unit change comparison plotted"
#| fig-cap-location: bottom

# Plotting
plt.rcParams.update({'font.size': 20})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# MAPE plot
ax1.bar(['Celsius', 'Kelvin'], [mape_celsius, mape_kelvin], color = ["#003DFD", "#b512b8"])
ax1.set_ylim(0, mape_celsius +0.5)
ax1.set_ylabel('MAPE (%)')
ax1.set_title('MAPE: Unit Sensitivity')

# WAPE plot
ax2.bar(['Celsius', 'Kelvin'], [wape_celsius, wape_kelvin], color = ["#003DFD", "#b512b8"])
ax2.set_ylim(0,mape_celsius +0.5)
ax2.set_ylabel('WAPE (%)')
ax2.set_title('WAPE: Unit Sensitivity')

# MAE plot
ax3.bar(['Celsius', 'Kelvin'], [mae_celsius, mae_kelvin], color = ["#003DFD", "#b512b8"])
ax3.set_ylabel('MAE')
ax3.set_title('MAE: Unit Invariance')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: SE error weight plot
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.15: Squared Error (SE) weight plot"
#| fig-cap-location: bottom

fig, ax = plt.subplots(figsize=(12, 6))

# Colours
colors = ["black", "#ff6933"]
# Generate data
error = np.linspace(-10, 10, 1000)
squared_error = error ** 2
# Plotting
ax.plot(error, squared_error, color=colors[0], linewidth=4)
ax.fill_between(error, 0, squared_error, color=colors[1], alpha=0.95)
# Labels etc
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Squared Error', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Limit y-axis 
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: AE and SE comparison  
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.16: Comparison of Absolute Error and Squared Error weightings"
#| fig-cap-location: bottom

actual = 5
forecasts = np.linspace(0, 10, 100)

# Calculate errors
absolute_error = np.abs(forecasts - actual)
squared_error = (forecasts - actual)**2

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecasts, absolute_error, label='Absolute Error', color='#003DFD', linewidth=3)
ax.plot(forecasts, squared_error, label='Squared Error', color='#b512b8', linewidth=3)

ax.set_xlabel('Forecast Value', fontsize=14)
ax.set_ylabel('Error', fontsize=14)
ax.set_title('Comparison of Absolute Error and Squared Error', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Scale invariance  
#| message: false
#| echo: true
#| eval: true
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

# ----------------------------------------------------------------------

#| label: Plot and accuracy of naive models 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.17: 3-Month Validation: Naive vs Seasonal Naive Forecast"
#| fig-cap-location: bottom
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

# Set up plot
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(15, 8))

# Color scheme
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Validation period
period_length = pd.Timedelta(days=90)

def naive_forecast(train_data, periods):
    return [train_data.iloc[-1]] * periods

def seasonal_naive_forecast(train_data, periods, season_length):
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
naive_forecast = naive_forecast(train_df['sold'], len(test_df))
seasonal_naive_forecast = seasonal_naive_forecast(train_df['sold'], len(test_df), 14)

# Calculate metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual+1, np.array(forecast)+1)
    return mae, rmse, mape

naive_metrics = calculate_metrics(test_df['sold'], naive_forecast)
seasonal_naive_metrics = calculate_metrics(test_df['sold'], seasonal_naive_forecast)

# Plot data and forecasts
ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=colors[0])
ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=colors[1])
ax.plot(test_df['date'], naive_forecast, label='Naive Forecast', color=colors[2], linestyle='--')
ax.plot(test_df['date'], seasonal_naive_forecast, label='Seasonal Naive Forecast', color=colors[4], linestyle='--')

# Highlight test period
ax.axvspan(test_start, test_end, alpha=0.2, color=colors[1])

# Labels etc
ax.set_ylabel('Sales')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
ax.grid(True)

# Add error metrics to plot
metric_names = ['MAE', 'RMSE', 'MAPE']
naive_text = '\n'.join([f'Naive {name}: {value:.2f}' for name, value in zip(metric_names, naive_metrics)])
seasonal_text = '\n'.join([f'Seasonal Naive {name}: {value:.2f}' for name, value in zip(metric_names, seasonal_naive_metrics)])

ax.text(0.02, 0.98, naive_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=colors[2])
ax.text(0.02, 0.84, seasonal_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=colors[4])

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Calculate error metrics
#| message: false
#| echo: true
#| eval: true
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual+1, np.array(forecast)+1)
    return mae, rmse, mape

def calculate_scaled_metrics(actual, forecast, training_data):
    in_sample_mae = np.mean(np.abs(np.diff(training_data)))
    if in_sample_mae == 0:
        return np.inf, np.inf
    errors = np.abs(actual - forecast)
    squared_errors = (actual - forecast) ** 2
    mase = np.mean(errors) / in_sample_mae
    rmsse = np.sqrt(np.mean(squared_errors)) / in_sample_mae
    return mase, rmsse

def calculate_relative_errors(actual, forecast, baseline_metrics):
    mae, rmse, mape = calculate_metrics(actual, forecast)
    rel_mae = mae / baseline_metrics[0] if baseline_metrics[0] != 0 else np.inf
    rel_rmse = rmse / baseline_metrics[1] if baseline_metrics[1] != 0 else np.inf
    rel_mape = mape / baseline_metrics[2] if baseline_metrics[2] != 0 else np.inf
    return rel_mae, rel_rmse, rel_mape

# ----------------------------------------------------------------------

#| label: Table of calculated error metrics
#| message: false
#| echo: false
#| eval: true
def print_metrics_table(title, metrics):
    print(f"\n{title}")
    print("-" * 50)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 50)
    for name, value in metrics.items():
        if np.isinf(value):
            print(f"{name:<20} {'inf':<15}")
        else:
            print(f"{name:<20} {value:<15.4f}")
    print("-" * 50)

# Calculate metrics for naive forecast
naive_mae, naive_rmse, naive_mape = calculate_metrics(test_df['sold'], naive_forecast)
naive_mase, naive_rmsse = calculate_scaled_metrics(test_df['sold'], naive_forecast, train_df['sold'])

# Calculate metrics for seasonal naive forecast
seasonal_mae, seasonal_rmse, seasonal_mape = calculate_metrics(test_df['sold'], seasonal_naive_forecast)
seasonal_mase, seasonal_rmsse = calculate_scaled_metrics(test_df['sold'], seasonal_naive_forecast, train_df['sold'])

# Calculate relative errors for seasonal naive forecast
seasonal_rel_mae, seasonal_rel_rmse, seasonal_rel_mape = calculate_relative_errors(
    test_df['sold'], seasonal_naive_forecast, (naive_mae, naive_rmse, naive_mape))

# Metric dictionaries
seasonal_metrics = {
    "MAE": seasonal_mae,
    "RMSE": seasonal_rmse,
    "MAPE": seasonal_mape,
    "MASE": seasonal_mase,
    "RMSSE": seasonal_rmsse,
    "Relative MAE": seasonal_rel_mae,
    "Relative RMSE": seasonal_rel_rmse,
    "Relative MAPE": seasonal_rel_mape
}

# Print
metrics_df = pd.DataFrame({
    'Metric': list(seasonal_metrics.keys()),
    'Value': list(seasonal_metrics.values())
})

# Format the values to 4 decimal places
metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}" if not np.isinf(x) else "inf")

# Display as HTML table
display(HTML(metrics_df.to_html(index=False, classes="table table-striped")))

# ----------------------------------------------------------------------

#| label: Error metrics applied to pooled data
#| message: false
#| echo: false
#| eval: true

# Example with three product sales series
product_actuals = {
    "ProductA": np.array([10, 12, 15, 13, 14]),
    "ProductB": np.array([21, 20, 22, 25, 24]),
    "ProductC": np.array([5, 3, 4, 6, 4])
}

product_forecasts = {
    "ProductA": np.array([11, 13, 14, 14, 15]),
    "ProductB": np.array([20, 21, 21, 23, 25]),
    "ProductC": np.array([4, 4, 5, 5, 5])
}

# Calculate metrics for individual products
for product in product_actuals:
    actuals = product_actuals[product]
    forecasts = product_forecasts[product]
    mae, rmse, mape = calculate_metrics(actuals, forecasts)
    print(f"{product} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")

# Pool all actuals and forecasts for group-level metrics
all_actuals = np.concatenate([product_actuals[k] for k in product_actuals])
all_forecasts = np.concatenate([product_forecasts[k] for k in product_forecasts])

# Calculate metrics on pooled data
pooled_mae, pooled_rmse, pooled_mape = calculate_metrics(all_actuals, all_forecasts)
print(f"\nPooled metrics - MAE: {pooled_mae:.2f}, RMSE: {pooled_rmse:.2f}, MAPE: {pooled_mape:.2f}")

# ----------------------------------------------------------------------

#| label: Forecast horizon length diagram
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.18: Impact of validation length on test set accruacy"
#| fig-cap-location: bottom

def error_metric_accuracy(x):
    x = np.array(x)
    return 1 - 0.9 * np.exp(-0.5 * x) + 0.1 * np.random.random(x.shape)

# Generate data
x = np.linspace(0, 10, 100)
y = error_metric_accuracy(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)

# Add labels and title
plt.xlabel('Validation Period Length', fontsize=12)
plt.ylabel('Error Metric Accuracy', fontsize=12)
plt.title('Impact of Validation Period Length on Error Metric Accuracy', fontsize=14)

# Customize x-axis
plt.xticks([0, 10], ['Short', 'Long'])

# Customize y-axis
plt.yticks([0, 1], ['Low', 'High'])

# Add annotations
plt.annotate('High variability', xy=(1, error_metric_accuracy(1)), xytext=(1, 0.3),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red')

plt.annotate('Increasing stability', xy=(5, error_metric_accuracy(5)), xytext=(5, 0.8),
             arrowprops=dict(facecolor='green', shrink=0.05), color='green')

plt.annotate('Diminishing returns', xy=(9, error_metric_accuracy(9)), xytext=(9, 0.9),
             arrowprops=dict(facecolor='purple', shrink=0.05), color='purple')

# Show grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Comparing forecast lengths plot 
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.19: Applied validation length changes on test set accruacy"
#| fig-cap-location: bottom

from dateutil.relativedelta import relativedelta
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

series_df['date'] = pd.to_datetime(series_df['date'])

# Let's take the last year of data
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

# Set up plot
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle('Effect of Validation Period Length on Forecast Accuracy', fontsize=20)

# Color scheme
colors = ["black", "#003DFD", "#b512b8"]

# Validation periods
validation_periods = [
    ('2 Weeks', 14),       # 14 days
    ('3 Months', 90),      # 90 days
    ('6 Months', 180)      # 180 days
]

# Simpler approach using more traditional forecasting methods
for i, (ax, (period_name, days)) in enumerate(zip(axs, validation_periods)):
    # Calculate split dates
    test_end_date = df['date'].max()
    test_start_date = test_end_date - pd.DateOffset(days=days)
    
    # Create split point
    train_df = df[df['date'] < test_start_date].copy()
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()
    
    # Simple exponential smoothing forecast
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Prepare time series for statsmodels
    train_ts = train_df.set_index('date')['sold']
    
    # Fit model with appropriate seasonality (7 days for weekly pattern)
    model = ExponentialSmoothing(
        train_ts, 
        seasonal_periods=7, 
        trend='add', 
        seasonal='add'
    ).fit()
    
    # Generate forecast
    forecast = model.forecast(len(test_df))
    
    # Make sure forecast index aligns with test dates
    forecast = pd.Series(
        forecast.values, 
        index=test_df['date']
    )
    
    # Calculate errors
    mape = mean_absolute_percentage_error(test_df['sold']+1, forecast.values+1)
    mae = mean_absolute_error(test_df['sold'], forecast.values)
    
    # Plot data
    ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
    ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=colors[0])
    ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=colors[1])
    ax.plot(test_df['date'], forecast.values, 
            label=f'Exp. Smoothing (MAE: {mae:.2f}, MAPE: {mape:.2%})', 
            color=colors[2], linestyle='--')
    
    # Highlight test period
    ax.axvspan(test_start_date, test_end_date, alpha=0.2, color=colors[1])
    
    # Labels etc
    ax.set_ylabel('Sales')
    ax.set_title(f'{period_name} Validation Period')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

# Set x-label
fig.text(0.5, 0.04, 'Date', ha='center', va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.show()

# ----------------------------------------------------------------------

#| label: Scaling data leakage
#| message: false
#| echo: false
#| eval: false

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import math

# Set random seed for reproducibility
np.random.seed(42)

# Load the air passengers dataset
df = pd.read_csv(r'C:\Users\Graeme\Documents\github\tsfwpt\data\passengers.csv', parse_dates=['date'])

# Sort by date just to be safe
df = df.sort_values('date').reset_index(drop=True)

# Define the split point: 36 months before the end
split_idx = len(df) - 48

# Separate into train/test
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Define color scheme
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# -----------------------------------------------------
# APPROACH 1: WRONG WAY (LEAKAGE) - Scale then split
# -----------------------------------------------------

# Create a copy of the data for the wrong approach
df_wrong = df.copy()

# Apply scaling to the entire dataset before splitting (LEAKAGE!)
scaler_wrong = MinMaxScaler(feature_range=(0, 1))
df_wrong['passengers_scaled'] = scaler_wrong.fit_transform(df_wrong[['passengers']])

# Split after scaling
train_wrong = df_wrong.iloc[:split_idx].copy()
test_wrong = df_wrong.iloc[split_idx:].copy()

# Fit ARIMA model on the wrongly scaled data
# Explicitly use statsmodels ARIMA to avoid confusion
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
model_wrong = StatsmodelsARIMA(train_wrong['passengers_scaled'], order=(12,1,0))
model_wrong_fit = model_wrong.fit()

# Generate forecasts
h = len(test_wrong)
fcst_wrong = model_wrong_fit.forecast(steps=h)

# Transform the forecasts back to the original scale
fcst_wrong_df = pd.DataFrame({'forecast_scaled': fcst_wrong})
fcst_wrong_df[['forecast_unscaled']] = scaler_wrong.inverse_transform(fcst_wrong_df[['forecast_scaled']])

# Calculate RMSE for the wrong approach
y_true = test_df['passengers'].values
y_pred_wrong = fcst_wrong_df['forecast_unscaled'].values
rmse_wrong = math.sqrt(mean_squared_error(y_true, y_pred_wrong))

# -----------------------------------------------------
# APPROACH 2: RIGHT WAY - Split then scale
# -----------------------------------------------------

# Create copies for the right approach
train_right = train_df.copy()
test_right = test_df.copy()

# Apply scaling only to training data
scaler_right = MinMaxScaler(feature_range=(0, 1))
train_right['passengers_scaled'] = scaler_right.fit_transform(train_right[['passengers']])

# Fit ARIMA model on the correctly scaled data
model_right = StatsmodelsARIMA(train_right['passengers_scaled'], order=(12,1,0))
model_right_fit = model_right.fit()

# Generate forecasts
fcst_right = model_right_fit.forecast(steps=h)

# Transform the forecasts back to the original scale
fcst_right_df = pd.DataFrame({'forecast_scaled': fcst_right})
fcst_right_df[['forecast_unscaled']] = scaler_right.inverse_transform(fcst_right_df[['forecast_scaled']])

# Calculate RMSE for the right approach
y_pred_right = fcst_right_df['forecast_unscaled'].values
rmse_right = math.sqrt(mean_squared_error(y_true, y_pred_right))

# -----------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------

plt.figure(figsize=(15, 8))

# Plot original data
plt.plot(df['date'], df['passengers'], color=colors[0], label='Actual Data')

# Plot train/test split
split_date = df.iloc[split_idx]['date']
plt.axvline(x=split_date, color='r', linestyle='--')

# Plot forecasts
plt.plot(test_df['date'], y_pred_wrong, color=colors[1], label='Forecast (Leaked Scaling)', linestyle='-', linewidth=2)
plt.plot(test_df['date'], y_pred_right, color=colors[2], label='Forecast (No Leakage)', linestyle='-', linewidth=2)

# Highlight training and test periods
plt.axvspan(df['date'].min(), split_date, alpha=0.1, color=colors[0])  # Training period
plt.axvspan(split_date, df['date'].max(), alpha=0.2, color=colors[1])  # Test period

# Add error metrics in a text box
textstr = '\n'.join((
    f'Leaked Scaling RMSE: {rmse_wrong:.2f}',
    f'Proper Scaling RMSE: {rmse_right:.2f}'
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.98, 0.05, textstr, transform=plt.gca().transAxes, fontsize=18,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Set labels and title with larger font
plt.title('Impact of Data Leakage in Scaling on Forecast Accuracy', fontsize=22)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Passengers', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend with larger font
plt.legend(fontsize=18)

# Apply grid and tight layout
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show plot
plt.show()

# ----------------------------------------------------------------------

#| label: Impact of splitting before and after data transformation
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.20: Impact of splitting before and after data transformation"
#| fig-cap-location: bottom

from sklearn.preprocessing import MinMaxScaler
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import LinearRegression
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.lines as mlines

# ----------------------------------------------------------------------
# 1) DATASET WITH TREND CHANGE
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2) WRONG DETRENDING (LEAKAGE)
# ----------------------------------------------------------------------
# Detrend entire series (knows about test part!)
df_wrong = df.copy()
detrended_full = signal.detrend(df_wrong['value'])
df_wrong['detrended'] = detrended_full

train_wrong = df_wrong[df_wrong['date'] < split_date].copy()
test_wrong  = df_wrong[df_wrong['date'] >= split_date].copy()

# Suppose we fit a model (e.g. ARIMA) on the wrongly-detrended train data
# and then forecast test data in the *detrended* space. We'll do a minimal example here:
model_wrong = ARIMA(train_wrong['detrended'], order=(7,1,7)).fit()
# Forecast in detrended space
fcast_wrong_detrended = model_wrong.forecast(len(test_wrong))

# Now add back the *full-data* linear trend that 'signal.detrend' removed. 
mean_diff_full = df_wrong['value'].mean() - df_wrong['detrended'].mean()
fcast_wrong = fcast_wrong_detrended + mean_diff_full

# ----------------------------------------------------------------------
# 3) CORRECT DETRENDING (NO LEAKAGE)
# ----------------------------------------------------------------------
# Fit trend *only* on training data
train_right = train_df.copy()
train_right['time_idx'] = np.arange(len(train_right))
X_train = train_right[['time_idx']]
y_train = train_right['value']

lr = LinearRegression().fit(X_train, y_train)
train_trend_pred = lr.predict(X_train)
train_right['detrended'] = train_right['value'] - train_trend_pred

# For test data, use *training* trend model
test_right = test_df.copy()
test_right['time_idx'] = np.arange(len(train_right), len(train_right)+len(test_right))
test_trend_pred = lr.predict(test_right[['time_idx']])
test_right['detrended'] = test_right['value'] - test_trend_pred

# Suppose we fit a model on the training detrended series
model_right = ARIMA(train_right['detrended'], order=(7,1,7)).fit()
fcast_right_detrended = model_right.forecast(len(test_right))

# Add back the *training-derived* trend
fcast_right = fcast_right_detrended + test_trend_pred

# ----------------------------------------------------------------------
# 4) VISUALIZATION WITH REQUESTED STYLING
# ----------------------------------------------------------------------

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_wrong = calculate_rmse(test_df['value'], fcast_wrong)
rmse_right = calculate_rmse(test_df['value'], fcast_right)

# Define color scheme
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Set up the plot
plt.figure(figsize=(12, 6))

# Create plot with consistent styling
plt.plot(df['date'], df['value'], color=colors[0], label='Original time series')
plt.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')
plt.plot(test_df['date'], fcast_wrong, color=colors[1], label='Forecast (Leaked Trend)', linestyle='--')
plt.plot(test_df['date'], fcast_right, color=colors[2], label='Forecast (No Leakage)', linestyle=':')

# Highlight training and test periods
plt.axvspan(df['date'].min(), split_date, alpha=0.1, color=colors[0])  # Training period
plt.axvspan(split_date, df['date'].max(), alpha=0.2, color=colors[1])  # Test period

# Enhance styling
plt.title(f'RMSE: Leaked Trend = {rmse_wrong:.2f}, No Leakage = {rmse_right:.2f}', fontsize=18)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Values', fontsize=20)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend with larger font
plt.legend(fontsize=16)


# Apply tight layout
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Impact of lookforward bias via lagged features
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.21: Impact of lookforward bias via lagged features"
#| fig-cap-location: bottom


# Generate sample data
np.random.seed(123)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100))  # Random walk
df = pd.DataFrame({'date': dates, 'value': values})

# Add lag features - this is fine for single-step prediction
df['lag1'] = df['value'].shift(1)
df['lag2'] = df['value'].shift(2)
df = df.dropna()

# Split data
split_idx = 80
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Train a simple model
from sklearn.linear_model import LinearRegression
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
# Plot comparisons
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Set up figure
plt.figure(figsize=(12, 8))

# First subplot - Wrong approach
plt.subplot(2, 1, 1)
plt.plot(train_df['date'], train_df['value'], color=colors[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=colors[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_wrong, color=colors[1], linestyle='--', label='Forecasts with Leakage')

# Highlight training and test periods

plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=colors[1])
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
plt.plot(train_df['date'], train_df['value'], color=colors[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=colors[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_right, color=colors[2], linestyle=':', label='Forecasts without Leakage')

# Highlight training and test periods

plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=colors[2])
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

# ----------------------------------------------------------------------

#| label: Target feature polluting predictions
#| message: false
#| echo: false
#| eval: true
#| fig-cap: "3.22: Impact of lookforward bias via golden feature"
#| fig-cap-location: bottom

# Sample panel data with multiple items
items = ['A', 'B', 'C']
panel_data = []

for item in items:
    # Create slightly different patterns for each item
    np.random.seed(ord(item))  # Different seed per item
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.normal(0, 1, 100)) + ord(item) - 65
    item_df = pd.DataFrame({
        'date': dates,
        'item_id': item,
        'sales': values
    })
    panel_data.append(item_df)

panel_df = pd.concat(panel_data)

# WRONG: Create aggregated features first, then split
panel_df_wrong = panel_df.copy()
# Create daily total sales across all items
daily_totals = panel_df_wrong.groupby('date')['sales'].sum().reset_index()
daily_totals.rename(columns={'sales': 'total_sales'}, inplace=True)

# Join back to create a feature (this now contains future information for each item!)
panel_df_wrong = panel_df_wrong.merge(daily_totals, on='date')

# Now split (too late, leakage occurred)
split_date = pd.to_datetime('2020-03-15')
train_wrong = panel_df_wrong[panel_df_wrong['date'] < split_date]
test_wrong = panel_df_wrong[panel_df_wrong['date'] >= split_date]

# CORRECT: Split first, then create aggregated features separately
panel_df_right = panel_df.copy()
# First split
train_right_base = panel_df_right[panel_df_right['date'] < split_date]
test_right_base = panel_df_right[panel_df_right['date'] >= split_date]

# Now create aggregated features only from training data
train_totals = train_right_base.groupby('date')['sales'].sum().reset_index()
train_totals.rename(columns={'sales': 'total_sales'}, inplace=True)

# Add to train set
train_right = train_right_base.merge(train_totals, on='date')

# For test set, if we need contemporaneous totals, we'd compute them
# from other series' predictions in production, not actuals
# For this example, just use the last training day's total
last_total = train_totals['total_sales'].iloc[-1]
test_right = test_right_base.copy()
test_right['total_sales'] = last_total  # In practice, this would use predictions

# Visualize the difference for one item
item_a_wrong = train_wrong[train_wrong['item_id'] == 'A']
item_a_test_wrong = test_wrong[test_wrong['item_id'] == 'A']
item_a_right = train_right[train_right['item_id'] == 'A']
item_a_test_right = test_right[test_right['item_id'] == 'A']

# Define color scheme
colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

# Set up figure
plt.figure(figsize=(12, 8))

# First subplot - Wrong approach
plt.subplot(2, 1, 1)
plt.plot(item_a_wrong['date'], item_a_wrong['total_sales'], color=colors[0], label='Training Total Sales (Wrong)')
plt.plot(item_a_test_wrong['date'], item_a_test_wrong['total_sales'], color=colors[1], label='Test Total Sales (Wrong)')

# Highlight training and test periods
plt.axvspan(split_date, item_a_test_wrong['date'].max(), alpha=0.2, color=colors[1])
plt.axvline(x=split_date, color='r', linestyle='--')
plt.title('WRONG: Aggregate Features Created Before Splitting (Leakage!)', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Total Sales', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Second subplot - Correct approach
plt.subplot(2, 1, 2)
plt.plot(item_a_right['date'], item_a_right['total_sales'], color=colors[0], label='Training Total Sales (Correct)')
plt.plot(item_a_test_right['date'], item_a_test_right['total_sales'], color=colors[2], label='Test Total Sales (Correct)')

# Highlight training and test periods
plt.axvspan(split_date, item_a_test_right['date'].max(), alpha=0.2, color=colors[2])
plt.axvline(x=split_date, color='r', linestyle='--')

plt.title('CORRECT: Aggregate Features Created Only From Training Data', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Total Sales', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Calculate the difference in test feature values
wrong_mean = item_a_test_wrong['total_sales'].mean()
right_mean = item_a_test_right['total_sales'].mean()
pct_diff = abs(wrong_mean - right_mean) / wrong_mean * 100

# Add a text annotation showing the magnitude of the leakage
plt.figtext(0.5, 0.01, 
            f'Impact of Leakage: Test feature values differ by {pct_diff:.1f}%', 
            ha='center', fontsize=14, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)  # Make room for the annotation
plt.show()