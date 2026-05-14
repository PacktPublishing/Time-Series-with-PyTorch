# =============================================================================
# Chapter 3: Evaluating Time-Series Models
# =============================================================================
# Code companion for Chapter 3
# Covers: bias-variance tradeoff, data partitioning, error metrics (AE, APE,
#         SE), extrinsic metrics (MASE, RMSSE, relative error), grouped error,
#         forecast horizon length, data leakage
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import math

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from cycler import cycler

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from scipy import signal
from dateutil.relativedelta import relativedelta
from IPython.display import display, HTML

warnings.filterwarnings("ignore")

# =============================================================================
# Setup: palette, rcParams, CFG
# =============================================================================

custom_palette = ["#000000", "#0072B2", "#D55E00", "#009E73",
                  "#CC79A7", "#56B4E9", "#E69F00"]
line_styles = ['-', '--', '-.', ':']

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18

class CFG:
    data_folder = Path.cwd() /  "Data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})

# =============================================================================
# Helper: darts-style plotting function
# =============================================================================

def darts_plot(df, x_column, y_columns, labels=None, quantiles=None, title=None, fontsize=18):
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))

    if labels is None:
        labels = y_columns

    alpha_confidence_intvls = 0.25
    darts_colors = custom_palette

    for i, y_col in enumerate(y_columns):
        color = darts_colors[i % len(darts_colors)]
        ax.plot(df[x_column], df[y_col], label=labels[i], color=color, linewidth=2)

        if quantiles is not None and y_col in quantiles:
            low_col, high_col = quantiles[y_col]
            low_series = df[low_col]
            high_series = df[high_col]

            if len(low_series) > 1:
                ax.fill_between(
                    df[x_column], low_series, high_series,
                    color=color, alpha=alpha_confidence_intvls,
                )
            else:
                ax.plot(
                    [df[x_column].iloc[0], df[x_column].iloc[0]],
                    [low_series.iloc[0], high_series.iloc[0]],
                    "-+", color=color, lw=2,
                )

    ax.set_xlabel(x_column, fontsize=fontsize)
    ax.set_ylabel('Value', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(prop={'size': fontsize})
    ax.set_facecolor('white')
    plt.tight_layout()
    return fig, ax


# =============================================================================
# Helper: seasonal decomposition plotting
# =============================================================================

def plot_seasonal_decompose(series, model='additive', figsize=(12, 10), fontsize=18):
    plt.rcParams.update({'font.size': fontsize})
    decomposition = seasonal_decompose(series, model=model)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)

    decomposition.observed.plot(ax=ax1, color=custom_palette[0], linewidth=0.5)
    ax1.set_ylabel('Observed', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax1.set_title('Observed Time Series', fontsize=fontsize)

    decomposition.trend.plot(ax=ax2, color=custom_palette[1], linewidth=1)
    ax2.set_ylabel('Trend', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.set_title('Trend Component', fontsize=fontsize)

    decomposition.seasonal.plot(ax=ax3, color=custom_palette[4], linewidth=0.5)
    ax3.set_ylabel('Seasonal', fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax3.set_title('Seasonal Component', fontsize=fontsize)

    ax4.scatter(decomposition.resid.index, decomposition.resid, color=custom_palette[5], s=3, alpha=0.5)
    ax4.set_ylabel('Residual', fontsize=fontsize)
    ax4.set_xlabel('Date', fontsize=fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax4.set_title('Residual Component', fontsize=fontsize)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    return fig


# =============================================================================
# 3.1  Bias-Variance Tradeoff
# =============================================================================

# --- Figure 3.1: Bias-variance tradeoff ---
x = np.linspace(0, 10, 100)
bias = 10 * np.exp(-0.5 * x)
variance = 0.05 * (x ** 2)
total_error = bias + variance

plt.figure(figsize=(10, 6))
plt.plot(x, bias, color=custom_palette[0], label='Bias²')
plt.plot(x, variance, color=custom_palette[1], label='Variance', linestyle='dashdot')
plt.plot(x, total_error, color=custom_palette[2], label='Total Error', linestyle=':')

optimal_x = (x[np.argmin(total_error)]) - 0.2
plt.axvline(x=optimal_x, color=custom_palette[3], linestyle='--', label='Optimal Model Complexity')

plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend(fontsize=12)
plt.text(0.5, 9, 'Bias²', color=custom_palette[0], fontsize=12)
plt.text(8, 2.8, 'Variance', color=custom_palette[1], fontsize=12)
plt.text(5, 3, 'Total Error', color=custom_palette[2], fontsize=12)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# --- Figure 3.2: Training and validation loss ---
iterations = np.linspace(0, 100, 1000)
training_loss = 1.6 * np.exp(-0.03 * iterations) + 0.1
validation_loss = 1.6 * np.exp(-0.03 * iterations) + 0.1 + 0.0002 * iterations**1.8

overfitting_point = iterations[np.argmin(validation_loss)]

plt.figure(figsize=(10, 6))
plt.plot(iterations, training_loss, color=custom_palette[0], label='Training loss', linewidth=2)
plt.plot(iterations, validation_loss, color=custom_palette[1], label='Validation loss', linewidth=2)
plt.axvline(x=overfitting_point, color=custom_palette[3], linestyle='--', linewidth=2)

plt.text(overfitting_point + 2, 1.2, 'Overfitting starts from here',
         rotation=0, verticalalignment='center', fontsize=12, color=custom_palette[3])

plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss over Iterations', fontsize=16)
plt.legend(fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(0, 1.6)
plt.tight_layout()
plt.show()


# =============================================================================
# 3.2  Comparison Data
# =============================================================================

# Load panel data and wrangle
panel_df = pd.read_csv(CFG.data_folder / 'M5_t20_ABC.csv', index_col=False).drop(columns='Unnamed: 0')
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

# Extract single series for analysis
series_df = panel_df.loc[panel_df.item_id == 'FOODS_2_197'].copy()
series_df['date'] = pd.to_datetime(series_df['date'])

# --- Figure 3.3: Product sales over time ---
fig, ax = darts_plot(series_df, 'date', ['sold'],
                     labels=['Sales'],
                     title='Sales for FOODS_2_197')
plt.show()


# --- Figure 3.4: Decomposition ---
series_df.set_index('date', inplace=True, drop=True)
short_series = series_df['2015':'2016'].copy()
fig = plot_seasonal_decompose(short_series['sold'], model='additive')
plt.show()


# =============================================================================
# 3.3  Stationarity Tests
# =============================================================================

# ADF test
result = adfuller(series_df['sold'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# KPSS test
kpss_stat, p_value, lags, crit = kpss(series_df['sold'])
print(f'KPSS Statistic: {kpss_stat}')
print(f'p-value: {p_value}')
print('Critical Values:')
for key, value in crit.items():
    print(f'\t{key}: {value}')


# --- Figure 3.5: Autocorrelation plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(series_df['sold'], lags=32, ax=ax1, color=custom_palette[1])
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(series_df['sold'], lags=32, ax=ax2, color=custom_palette[1])
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()


# --- Figure 3.6: Residuals from OLS ---
series_df.reset_index(inplace=True)
series_df['days'] = (pd.to_datetime(series_df['date']) - pd.to_datetime(series_df['date'].min())).dt.days

model = LinearRegression()
X = series_df[['days']]
y = series_df['sold']
model.fit(X, y)
series_df['predicted'] = model.predict(X)
series_df['residuals'] = series_df['sold'] - series_df['predicted']

plt.figure(figsize=(12, 6))
plt.scatter(series_df['date'], series_df['residuals'], alpha=0.5, color=custom_palette[0])
plt.axhline(y=0, color=custom_palette[1], linestyle='-', linewidth=2)
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =============================================================================
# 3.4  Fixed-Origin Designs
# =============================================================================

# Simple train-test split
from dateutil.relativedelta import relativedelta
last_date = series_df['date'].max()
split_date = last_date - relativedelta(months=3)

train_df = series_df[series_df['date'] < split_date]
test_df = series_df[series_df['date'] >= split_date]

# --- Figure 3.7: Simple train-test split ---
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color=custom_palette[0])
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color=custom_palette[1])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()


# Train-validation-test split
last_date = series_df['date'].max()
test_split_date = last_date - relativedelta(months=3)
val_split_date = test_split_date - relativedelta(months=3)

train_df = series_df[series_df['date'] < val_split_date]
val_df = series_df[(series_df['date'] >= val_split_date) & (series_df['date'] < test_split_date)]
test_df = series_df[series_df['date'] >= test_split_date]

# --- Figure 3.8: Train-validation-test split ---
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['sold'], label='Training Data', color=custom_palette[0])
plt.plot(val_df['date'], val_df['sold'], label='Validation Data', color=custom_palette[4])
plt.plot(test_df['date'], test_df['sold'], label='Test Data', color=custom_palette[1])
plt.tight_layout()
plt.title('Train-Validation-Test Split')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 3.5  Cross-Validation
# =============================================================================

# --- Figure 3.9: Expanding Window CV ---
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

last_date = series_df['date'].max()
test_periods = [relativedelta(months=3 * i) for i in range(3, 0, -1)]

train_line = mlines.Line2D([], [], color=custom_palette[0], label='Training Data')
test_line = mlines.Line2D([], [], color=custom_palette[1], label='Test Data')
excluded_line = mlines.Line2D([], [], color='gray', alpha=0.5, label='Excluded Data')
legend_items = [train_line, test_line, excluded_line]

for fold, (ax, test_period) in enumerate(zip(reversed(axs), test_periods), 1):
    test_end = last_date
    test_start = test_end - relativedelta(months=3)
    train_end = test_start

    train_df = series_df[series_df['date'] < train_end]
    test_df = series_df[(series_df['date'] >= test_start) & (series_df['date'] < test_end)]
    excluded_df = series_df[series_df['date'] >= test_end]

    ax.plot(train_df['date'], train_df['sold'], color=custom_palette[0])
    ax.plot(test_df['date'], test_df['sold'], color=custom_palette[1])
    if not excluded_df.empty:
        ax.plot(excluded_df['date'], excluded_df['sold'], color='gray', alpha=0.5)

    ax.axvspan(test_start, test_end, alpha=0.2, color=custom_palette[1])
    ax.set_ylabel('Sales')
    ax.set_title(f'Fold {4-fold}')
    ax.grid(True)

    last_date = test_start

fig.text(0.5, 0.01, 'Date', ha='center', va='center')
fig.legend(handles=legend_items, loc='upper center',
           bbox_to_anchor=(0.5, 0.98), ncol=3)
plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()


# --- Figure 3.10: Rolling Window CV ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

last_date = series_df['date'].max()
test_periods = [relativedelta(months=3 * i) for i in range(3, 0, -1)]

train_line = mlines.Line2D([], [], color=custom_palette[0], label='Training Data')
test_line = mlines.Line2D([], [], color=custom_palette[1], label='Test Data')
excluded_line = mlines.Line2D([], [], color='gray', alpha=0.5, label='Excluded Data')
legend_items = [train_line, test_line, excluded_line]

for fold, (ax, test_period) in enumerate(zip(reversed(axs), test_periods), 1):
    test_end = last_date
    test_start = test_end - relativedelta(months=3)
    train_end = test_start
    train_start = train_end - relativedelta(years=2)

    train_df = series_df[(series_df['date'] >= train_start) & (series_df['date'] < train_end)]
    test_df = series_df[(series_df['date'] >= test_start) & (series_df['date'] < test_end)]
    excluded_before_df = series_df[series_df['date'] < train_start]
    excluded_after_df = series_df[series_df['date'] >= test_end]

    if not excluded_before_df.empty:
        ax.plot(excluded_before_df['date'], excluded_before_df['sold'], color='gray', alpha=0.5)
    ax.plot(train_df['date'], train_df['sold'], color=custom_palette[0])
    ax.plot(test_df['date'], test_df['sold'], color=custom_palette[1])
    if not excluded_after_df.empty:
        ax.plot(excluded_after_df['date'], excluded_after_df['sold'], color='gray', alpha=0.5)

    ax.axvspan(train_start, train_end, alpha=0.1, color=custom_palette[0])
    ax.axvspan(test_start, test_end, alpha=0.2, color=custom_palette[1])
    ax.set_ylabel('Sales', fontsize=20)
    ax.set_title(f'Fold {4-fold}', fontsize=20)
    ax.grid(True)

    last_date = test_start

fig.text(0.53, 0.01, 'Date', ha='center', va='center', fontsize=20)
fig.legend(handles=legend_items, loc='upper center',
           bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=18)
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()


# =============================================================================
# 3.6  Error Metrics — Absolute Error (AE)
# =============================================================================

# --- Figure 3.11: Absolute error weighting ---
fig, ax = plt.subplots(figsize=(12, 6))
error = np.linspace(-100, 100, 1000)
absolute_error = np.abs(error)
ax.plot(error, absolute_error, color=custom_palette[0], linewidth=4)
ax.fill_between(error, 0, absolute_error, color=custom_palette[2], alpha=0.95)
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Absolute Error', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# =============================================================================
# 3.7  Error Metrics — Absolute Percentage Error (APE)
# =============================================================================

# --- Figure 3.12: Model comparison with MAPE ---
t = np.linspace(0, 2, 3)
yt_actuals = 0.01 + 0.1 * t
yt_A = 0.02 + 0.1 * t
yt_B = 0.01005 + 0.135 * t + 0.005 * t**2

plt.figure(figsize=(12, 6))
plt.plot(t, yt_actuals, label='Actuals', color=custom_palette[0], linewidth=3)
plt.plot(t, yt_A, label='A: Good model', color=custom_palette[1], linestyle=':', linewidth=3)
plt.plot(t, yt_B, label='B: Poor model', color=custom_palette[4], linestyle='--', linewidth=3)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Y value', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()


# MAPE calculation
def mape(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

mape_A = mape(yt_actuals, yt_A)
mape_B = mape(yt_actuals, yt_B)

print(f'Model A has a MAPE of {mape_A}')
print(f'Model B has a MAPE score of {mape_B}')


# Shifted MAPE to mitigate near-zero issue
mape_A = mape(yt_actuals + 1, yt_A + 1)
mape_B = mape(yt_actuals + 1, yt_B + 1)

print(f'Model A has a shifted MAPE of {mape_A}')
print(f'Model B have a shifted MAPE score of {mape_B}')


# --- Figure 3.13: APE error weighting ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Absolute Error', fontsize=20)

actuals = np.linspace(1, 10, 100)
forecasts = np.linspace(10, 1, 100)
percent_error = np.abs((actuals - forecasts) / actuals) * 100

ax.plot(actuals, percent_error, color=custom_palette[0], linewidth=4)
ax.fill_between(actuals, 0, percent_error, color=custom_palette[2], alpha=0.95)
ax.set_xlabel('Actuals', fontsize=18)
ax.set_ylabel('Absolute Percent Error', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

x_ticks = np.arange(1, 11, 1)
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'A:{a}\nF:{f}' for a, f in zip(x_ticks, x_ticks[::-1])], fontsize=18)
ax.set_ylim(0, 900)
plt.tight_layout()
plt.show()


# APE unit sensitivity: Celsius vs Kelvin
actual_celsius = np.array([20, 25, 30, 35, 40])
forecast_celsius = np.array([22, 24, 31, 33, 41])

actual_kelvin = actual_celsius + 273.15
forecast_kelvin = forecast_celsius + 273.15

def wape(actual, forecast):
    return np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual)) * 100

def mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))

mape_celsius = mape(actual_celsius, forecast_celsius)
mape_kelvin = mape(actual_kelvin, forecast_kelvin)
wape_celsius = wape(actual_celsius, forecast_celsius)
wape_kelvin = wape(actual_kelvin, forecast_kelvin)
mae_celsius = mae(actual_celsius, forecast_celsius)
mae_kelvin = mae(actual_kelvin, forecast_kelvin)

metrics = {
    'Metric': ['MAPE', 'WAPE', 'MAE'],
    'Celsius': [mape_celsius, wape_celsius, mae_celsius],
    'Kelvin': [mape_kelvin, wape_kelvin, mae_kelvin]
}
metrics_df = pd.DataFrame(metrics)
for col in ['Celsius', 'Kelvin']:
    metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}{'%' if x != mae_celsius and x != mae_kelvin else ''}")

print(metrics_df)



# --- Figure 3.14: APE unit change comparison ---
plt.rcParams.update({'font.size': 20})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.bar(['Celsius', 'Kelvin'], [mape_celsius, mape_kelvin], color=[custom_palette[1], custom_palette[4]])
ax1.set_ylim(0, mape_celsius + 0.5)
ax1.set_ylabel('MAPE (%)')
ax1.set_title('MAPE: Unit Sensitivity')

ax2.bar(['Celsius', 'Kelvin'], [wape_celsius, wape_kelvin], color=[custom_palette[1], custom_palette[4]])
ax2.set_ylim(0, mape_celsius + 0.5)
ax2.set_ylabel('WAPE (%)')
ax2.set_title('WAPE: Unit Sensitivity')

ax3.bar(['Celsius', 'Kelvin'], [mae_celsius, mae_kelvin], color=[custom_palette[1], custom_palette[4]])
ax3.set_ylabel('MAE')
ax3.set_title('MAE: Unit Invariance')

plt.tight_layout()
plt.show()


# =============================================================================
# 3.8  Error Metrics — Squared Error (SE)
# =============================================================================

# --- Figure 3.15: SE error weighting ---
fig, ax = plt.subplots(figsize=(12, 6))
error = np.linspace(-10, 10, 1000)
squared_error = error ** 2
ax.plot(error, squared_error, color=custom_palette[0], linewidth=4)
ax.fill_between(error, 0, squared_error, color=custom_palette[2], alpha=0.95)
ax.set_xlabel('Error', fontsize=18)
ax.set_ylabel('Squared Error', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()


# --- Figure 3.16: AE vs SE comparison ---
actual = 5
forecasts = np.linspace(0, 10, 100)
absolute_error = np.abs(forecasts - actual)
squared_error = (forecasts - actual)**2

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecasts, absolute_error, label='Absolute Error', color=custom_palette[1], linewidth=3)
ax.plot(forecasts, squared_error, label='Squared Error', color=custom_palette[4], linewidth=3)
ax.set_xlabel('Forecast Value', fontsize=14)
ax.set_ylabel('Error', fontsize=14)
ax.set_title('Comparison of Absolute Error and Squared Error', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# MSE scale invariance demonstration
actual = np.array([100, 150, 200, 250, 300])
forecast = np.array([105, 160, 190, 240, 310])

def mse(actual, forecast):
    return np.mean((actual - forecast)**2)

mse_original = mse(actual, forecast)
mse_scaled = mse(actual * 10, forecast * 10)

print(f"MSE (original): {mse_original:.2f}")
print(f"MSE (scaled): {mse_scaled:.2f}")
print(f"Ratio of scaled to original MSE: {mse_scaled / mse_original:.2f}")


# =============================================================================
# 3.9  Extrinsic Metrics — Naive Forecasts
# =============================================================================

# Take the last year of data
series_df['date'] = pd.to_datetime(series_df['date'])
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(15, 8))

period_length = pd.Timedelta(days=90)

def naive_forecast(train_data, periods):
    return [train_data.iloc[-1]] * periods

def seasonal_naive_forecast(train_data, periods, season_length):
    forecast = []
    for i in range(periods):
        forecast.append(train_data.iloc[-(season_length - i % season_length)])
    return forecast

test_end = df['date'].max()
test_start = test_end - period_length
train_end = test_start

train_df = df[df['date'] < train_end]
test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]

naive_fc = naive_forecast(train_df['sold'], len(test_df))
seasonal_naive_fc = seasonal_naive_forecast(train_df['sold'], len(test_df), 14)

# --- Figure 3.17: Naive vs Seasonal Naive ---
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape_val = mean_absolute_percentage_error(actual+1, np.array(forecast)+1)
    return mae, rmse, mape_val

naive_metrics = calculate_metrics(test_df['sold'], naive_fc)
seasonal_naive_metrics = calculate_metrics(test_df['sold'], seasonal_naive_fc)

ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=custom_palette[0])
ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=custom_palette[1])
ax.plot(test_df['date'], naive_fc, label='Naive Forecast', color=custom_palette[4], linestyle='--')
ax.plot(test_df['date'], seasonal_naive_fc, label='Seasonal Naive Forecast', color=custom_palette[3], linestyle='--')

ax.axvspan(test_start, test_end, alpha=0.2, color=custom_palette[1])
ax.set_ylabel('Sales')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
ax.grid(True)

metric_names = ['MAE', 'RMSE', 'MAPE']
naive_text = '\n'.join([f'Naive {name}: {value:.2f}' for name, value in zip(metric_names, naive_metrics)])
seasonal_text = '\n'.join([f'Seasonal Naive {name}: {value:.2f}' for name, value in zip(metric_names, seasonal_naive_metrics)])

ax.text(0.02, 0.98, naive_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=custom_palette[4])
ax.text(0.02, 0.84, seasonal_text, transform=ax.transAxes, verticalalignment='top', fontsize=18, color=custom_palette[3])

plt.tight_layout()
plt.show()


# =============================================================================
# 3.10  Scaled and Relative Metrics
# =============================================================================

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
    mae_val, rmse_val, mape_val = calculate_metrics(actual, forecast)
    rel_mae = mae_val / baseline_metrics[0] if baseline_metrics[0] != 0 else np.inf
    rel_rmse = rmse_val / baseline_metrics[1] if baseline_metrics[1] != 0 else np.inf
    rel_mape = mape_val / baseline_metrics[2] if baseline_metrics[2] != 0 else np.inf
    return rel_mae, rel_rmse, rel_mape


# Calculate all metrics for seasonal naive forecast
naive_mae, naive_rmse, naive_mape = calculate_metrics(test_df['sold'], naive_fc)
naive_mase, naive_rmsse = calculate_scaled_metrics(test_df['sold'], naive_fc, train_df['sold'])

seasonal_mae, seasonal_rmse, seasonal_mape = calculate_metrics(test_df['sold'], seasonal_naive_fc)
seasonal_mase, seasonal_rmsse = calculate_scaled_metrics(test_df['sold'], seasonal_naive_fc, train_df['sold'])

seasonal_rel_mae, seasonal_rel_rmse, seasonal_rel_mape = calculate_relative_errors(
    test_df['sold'], seasonal_naive_fc, (naive_mae, naive_rmse, naive_mape))

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

metrics_df = pd.DataFrame({
    'Metric': list(seasonal_metrics.keys()),
    'Value': list(seasonal_metrics.values())
})
metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}" if not np.isinf(x) else "inf")
print(metrics_df)


# =============================================================================
# 3.11  Grouped Error Metrics
# =============================================================================

# Pooled metrics across multiple products
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

for product in product_actuals:
    actuals_p = product_actuals[product]
    forecasts_p = product_forecasts[product]
    mae_val, rmse_val, mape_val = calculate_metrics(actuals_p, forecasts_p)
    print(f"{product} - MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}, MAPE: {mape_val:.2f}")

all_actuals = np.concatenate([product_actuals[k] for k in product_actuals])
all_forecasts = np.concatenate([product_forecasts[k] for k in product_forecasts])
pooled_mae, pooled_rmse, pooled_mape = calculate_metrics(all_actuals, all_forecasts)
print(f"\nPooled metrics - MAE: {pooled_mae:.2f}, RMSE: {pooled_rmse:.2f}, MAPE: {pooled_mape:.2f}")


# =============================================================================
# 3.12  Length of Evaluation
# =============================================================================

# --- Figure 3.18: Impact of validation length (conceptual) ---
def error_metric_accuracy(x):
    x = np.array(x)
    return 1 - 0.9 * np.exp(-0.5 * x) + 0.1 * np.random.random(x.shape)

x = np.linspace(0, 10, 100)
y = error_metric_accuracy(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color=custom_palette[1], linewidth=2)
plt.xlabel('Validation Period Length', fontsize=12)
plt.ylabel('Error Metric Accuracy', fontsize=12)
plt.title('Impact of Validation Period Length on Error Metric Accuracy', fontsize=14)
plt.xticks([0, 10], ['Short', 'Long'])
plt.yticks([0, 1], ['Low', 'High'])
plt.annotate('High variability', xy=(1, error_metric_accuracy(1)-0.005), xytext=(1, 0.3),
             arrowprops=dict(facecolor=custom_palette[2], shrink=0.05), color=custom_palette[2])
plt.annotate('Increasing stability', xy=(5, error_metric_accuracy(5)-0.005), xytext=(5, 0.8),
             arrowprops=dict(facecolor=custom_palette[3], shrink=0.05), color=custom_palette[3])
plt.annotate('Diminishing returns', xy=(9, error_metric_accuracy(9)-0.005), xytext=(9, 0.9),
             arrowprops=dict(facecolor=custom_palette[4], shrink=0.1), color=custom_palette[4])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- Figure 3.19: Applied validation length changes ---
series_df['date'] = pd.to_datetime(series_df['date'])
last_year = series_df['date'].max() - pd.DateOffset(years=1)
df = series_df[series_df['date'] > last_year].copy()

fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle('Effect of Validation Period Length on Forecast Accuracy', fontsize=20)

validation_periods = [
    ('2 Weeks', 14),
    ('3 Months', 90),
    ('6 Months', 180)
]

for i, (ax, (period_name, days)) in enumerate(zip(axs, validation_periods)):
    test_end_date = df['date'].max()
    test_start_date = test_end_date - pd.DateOffset(days=days)

    train_df = df[df['date'] < test_start_date].copy()
    test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()

    train_ts = train_df.set_index('date')['sold']

    model = ExponentialSmoothing(
        train_ts,
        seasonal_periods=7,
        trend='add',
        seasonal='add'
    ).fit()

    forecast = model.forecast(len(test_df))
    forecast = pd.Series(forecast.values, index=test_df['date'])

    mape_val = mean_absolute_percentage_error(test_df['sold']+1, forecast.values+1)
    mae_val = mean_absolute_error(test_df['sold'], forecast.values)

    ax.plot(df['date'], df['sold'], label='Full Data', color='gray', alpha=0.5)
    ax.plot(train_df['date'], train_df['sold'], label='Training Data', color=custom_palette[0])
    ax.plot(test_df['date'], test_df['sold'], label='Validation Data', color=custom_palette[1])
    ax.plot(test_df['date'], forecast.values,
            label=f'Exp. Smoothing (MAE: {mae_val:.2f}, MAPE: {mape_val:.2%})',
            color=custom_palette[4], linestyle='--')

    ax.axvspan(test_start_date, test_end_date, alpha=0.2, color=custom_palette[1])
    ax.set_ylabel('Sales')
    ax.set_title(f'{period_name} Validation Period')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

fig.text(0.5, 0.04, 'Date', ha='center', va='center')
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.show()


# =============================================================================
# 3.13  Data Leakage — Detrending
# =============================================================================

# --- Figure 3.20: Impact of leaked vs correct detrending ---
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
split_idx = 270
split_date = dates[split_idx]

train_indices = np.arange(split_idx)
test_indices = np.arange(split_idx, 365)

train_trend = 10 + 0.05 * train_indices + 0.0002 * train_indices**2
test_trend_start = train_trend[-1]
test_trend = test_trend_start - 0.1*(test_indices - test_indices[0]) \
                             - 0.001*(test_indices - test_indices[0])**2
trend = np.concatenate([train_trend, test_trend])
seasonal = 5 * np.sin(np.arange(365) * 2*np.pi/7)
noise = np.random.normal(0, 1, 365)
values = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'value': values})
train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

# Wrong: detrend entire series
df_wrong = df.copy()
detrended_full = signal.detrend(df_wrong['value'])
df_wrong['detrended'] = detrended_full

train_wrong = df_wrong[df_wrong['date'] < split_date].copy()
test_wrong = df_wrong[df_wrong['date'] >= split_date].copy()

model_wrong = ARIMA(train_wrong['detrended'], order=(7,1,7)).fit()
fcast_wrong_detrended = model_wrong.forecast(len(test_wrong))
mean_diff_full = df_wrong['value'].mean() - df_wrong['detrended'].mean()
fcast_wrong = fcast_wrong_detrended + mean_diff_full

# Correct: detrend only training data
train_right = train_df.copy()
train_right['time_idx'] = np.arange(len(train_right))
X_train = train_right[['time_idx']]
y_train = train_right['value']

lr = LinearRegression().fit(X_train, y_train)
train_trend_pred = lr.predict(X_train)
train_right['detrended'] = train_right['value'] - train_trend_pred

test_right = test_df.copy()
test_right['time_idx'] = np.arange(len(train_right), len(train_right)+len(test_right))
test_trend_pred = lr.predict(test_right[['time_idx']])
test_right['detrended'] = test_right['value'] - test_trend_pred

model_right = ARIMA(train_right['detrended'], order=(7,1,7)).fit()
fcast_right_detrended = model_right.forecast(len(test_right))
fcast_right = fcast_right_detrended + test_trend_pred

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_wrong = calculate_rmse(test_df['value'], fcast_wrong)
rmse_right = calculate_rmse(test_df['value'], fcast_right)

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['value'], color=custom_palette[0], label='Original time series')
plt.axvline(x=split_date, color=custom_palette[2], linestyle='--', label='Train/Test Split')
plt.plot(test_df['date'], fcast_wrong, color=custom_palette[1], label='Forecast (Leaked Trend)', linestyle='--')
plt.plot(test_df['date'], fcast_right, color=custom_palette[4], label='Forecast (No Leakage)', linestyle=':')
plt.axvspan(df['date'].min(), split_date, alpha=0.1, color=custom_palette[0])
plt.axvspan(split_date, df['date'].max(), alpha=0.2, color=custom_palette[1])
plt.title(f'RMSE: Leaked Trend = {rmse_wrong:.2f}, No Leakage = {rmse_right:.2f}', fontsize=18)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Values', fontsize=20)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# 3.14  Data Leakage — Lookforward Bias (Lagged Features)
# =============================================================================

# --- Figure 3.21: Lookforward bias via lagged features ---
np.random.seed(123)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100))
df = pd.DataFrame({'date': dates, 'value': values})

df['lag1'] = df['value'].shift(1)
df['lag2'] = df['value'].shift(2)
df = df.dropna()

split_idx = 80
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

model = LinearRegression()
X_train = train_df[['lag1', 'lag2']]
y_train = train_df['value']
model.fit(X_train, y_train)

# Wrong: actual lag values in test set
forecasts_wrong = []
for i in range(len(test_df)):
    X_test = test_df.iloc[i:i+1][['lag1', 'lag2']]
    pred = model.predict(X_test)[0]
    forecasts_wrong.append(pred)

# Correct: recursive forecasting
forecasts_right = []
last_values = train_df['value'].tail(2).values
for i in range(len(test_df)):
    X_test = np.array([[last_values[-1], last_values[-2]]])
    pred = model.predict(X_test)[0]
    forecasts_right.append(pred)
    last_values = np.append(last_values[1:], pred)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(train_df['date'], train_df['value'], color=custom_palette[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=custom_palette[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_wrong, color=custom_palette[1], linestyle='--', label='Forecasts with Leakage')
plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=custom_palette[1])
plt.axvline(x=test_df['date'].min(), color=custom_palette[2], linestyle='--')
plt.title('WRONG: Using Actual Values for Lagged Features (Leakage!)', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(train_df['date'], train_df['value'], color=custom_palette[0], label='Training Data')
plt.plot(test_df['date'], test_df['value'], color=custom_palette[3], label='Actual Values')
plt.plot(test_df['date'], forecasts_right, color=custom_palette[4], linestyle=':', label='Forecasts without Leakage')
plt.axvspan(test_df['date'].min(), test_df['date'].max(), alpha=0.2, color=custom_palette[4])
plt.axvline(x=test_df['date'].min(), color=custom_palette[2], linestyle='--')
plt.title('CORRECT: Recursive Forecasting Using Only Predictions', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

rmse_wrong = np.sqrt(mean_squared_error(test_df['value'], forecasts_wrong))
rmse_right = np.sqrt(mean_squared_error(test_df['value'], forecasts_right))

plt.figtext(0.5, 0.01, f'RMSE: Leaked Features = {rmse_wrong:.2f}, No Leakage = {rmse_right:.2f}',
            ha='center', fontsize=14, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()


# =============================================================================
# 3.15  Data Leakage — Panel Data (Golden Feature)
# =============================================================================

# --- Figure 3.22: Panel data leakage via golden feature ---
items = ['A', 'B', 'C']
panel_data = []

for item in items:
    np.random.seed(ord(item))
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.normal(0, 1, 100)) + ord(item) - 65
    item_df = pd.DataFrame({
        'date': dates,
        'item_id': item,
        'sales': values
    })
    panel_data.append(item_df)

panel_df = pd.concat(panel_data)

# Wrong: aggregate before splitting
panel_df_wrong = panel_df.copy()
daily_totals = panel_df_wrong.groupby('date')['sales'].sum().reset_index()
daily_totals.rename(columns={'sales': 'total_sales'}, inplace=True)
panel_df_wrong = panel_df_wrong.merge(daily_totals, on='date')

split_date = pd.to_datetime('2020-03-15')
train_wrong = panel_df_wrong[panel_df_wrong['date'] < split_date]
test_wrong = panel_df_wrong[panel_df_wrong['date'] >= split_date]

# Correct: split first, then aggregate
panel_df_right = panel_df.copy()
train_right_base = panel_df_right[panel_df_right['date'] < split_date]
test_right_base = panel_df_right[panel_df_right['date'] >= split_date]

train_totals = train_right_base.groupby('date')['sales'].sum().reset_index()
train_totals.rename(columns={'sales': 'total_sales'}, inplace=True)
train_right = train_right_base.merge(train_totals, on='date')

last_total = train_totals['total_sales'].iloc[-1]
test_right = test_right_base.copy()
test_right['total_sales'] = last_total

# Visualise for item A
item_a_wrong = train_wrong[train_wrong['item_id'] == 'A']
item_a_test_wrong = test_wrong[test_wrong['item_id'] == 'A']
item_a_right = train_right[train_right['item_id'] == 'A']
item_a_test_right = test_right[test_right['item_id'] == 'A']

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(item_a_wrong['date'], item_a_wrong['total_sales'], color=custom_palette[0], label='Training Total Sales (Wrong)')
plt.plot(item_a_test_wrong['date'], item_a_test_wrong['total_sales'], color=custom_palette[1], label='Test Total Sales (Wrong)')
plt.axvspan(split_date, item_a_test_wrong['date'].max(), alpha=0.2, color=custom_palette[1])
plt.axvline(x=split_date, color=custom_palette[2], linestyle='--')
plt.title('WRONG: Aggregate Features Created Before Splitting (Leakage!)', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Total Sales', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(item_a_right['date'], item_a_right['total_sales'], color=custom_palette[0], label='Training Total Sales (Correct)')
plt.plot(item_a_test_right['date'], item_a_test_right['total_sales'], color=custom_palette[4], label='Test Total Sales (Correct)')
plt.axvspan(split_date, item_a_test_right['date'].max(), alpha=0.2, color=custom_palette[4])
plt.axvline(x=split_date, color=custom_palette[2], linestyle='--')
plt.title('CORRECT: Aggregate Features Created Only From Training Data', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Total Sales', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

wrong_mean = item_a_test_wrong['total_sales'].mean()
right_mean = item_a_test_right['total_sales'].mean()
pct_diff = abs(wrong_mean - right_mean) / wrong_mean * 100

plt.figtext(0.5, 0.01,
            f'Impact of Leakage: Test feature values differ by {pct_diff:.1f}%',
            ha='center', fontsize=14, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()