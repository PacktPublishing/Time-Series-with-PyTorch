# Extracted from chapter17_unsupervised_anomly.qmd
# Do not edit the source .qmd file directly.

#| label: libraries and graph set-up
#| message: false
#| echo: false
#| eval: true

from pathlib import Path 

import pandas as pd
import numpy as np
import warnings
import itertools

import matplotlib.pyplot as plt
import seaborn as sns  

custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]

# general settings
class CFG:
    data_folder = Path.cwd().parent / "das_buch"/"data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18
    
    
# plotting parameters 
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18    
plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)})

# ----------------------------------------------------------------------

#| label: load pseudo sales data
#| message: false
#| echo: false
#| eval: true

file_path = CFG.data_folder / "chapter17" / "pseudo_sales_adj_v1.csv"

data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data.set_index('date', inplace=True)
data = data.sort_index()

# ----------------------------------------------------------------------

#| label: initial sales plot
#| fig-cap: "Figure 17.3: Pseudo sales data with no anomalies, showing two Christmas peaks and promotional-regime."
#| fig-cap-location: bottom
#| fig-alt: "Line plot of daily sales across roughly two years, with seasonal peaks in December."
#| message: false
#| echo: false
#| eval: true

# Create plot 
fig, ax = plt.subplots()
ax.plot(data.index, data['sales'], color=custom_palette[0], label='Sales')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: create point outlier and anomaly
#| fig-cap: "17.4: Sales data with point outlier and contextual anomaly"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

# Create point outlier and anomaly based on rolling std
point_outlier = pd.Timestamp('2022-10-04')
rolling_std_dev = data['sales'].rolling(window=14).std()
std_dev_rolling = rolling_std_dev.loc[point_outlier - pd.Timedelta(days=1)]
data.loc[point_outlier, 'sales'] = (
    data.loc[point_outlier - pd.Timedelta(days=1), 'sales'] + (5 * std_dev_rolling)
)

point_anomaly = pd.Timestamp('2021-08-27') # 2021-07-27
std_dev_rolling = rolling_std_dev.loc[point_anomaly - pd.Timedelta(days=1)]
data.loc[point_anomaly, 'sales'] = (
    data.loc[point_anomaly - pd.Timedelta(days=1), 'sales'] + (1.2 * std_dev_rolling)
)

# Rerun plot
fig, ax = plt.subplots()
ax.plot(data.index, data['sales'], color=custom_palette[0], label='Sales')
ax.scatter(
    point_outlier,
    data.loc[point_outlier, 'sales'],
    color=custom_palette[2], marker='o', s=100, label='Point Outlier', zorder=5
)
ax.scatter(
    point_anomaly,
    data.loc[point_anomaly, 'sales'],
    color=custom_palette[2], marker='x', s=100, label='Point Anomaly',zorder=5
)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: save point anomaly labels
#| message: false
#| echo: true
#| eval: true

data_pointanoms = data.copy()
data_pointanoms['is_anomaly'] = 0
outlier_dates = [point_outlier + pd.Timedelta(days=i) for i in range(1)]
anomaly_dates = [point_anomaly + pd.Timedelta(days=i) for i in range(1)]

data_pointanoms.loc[outlier_dates, 'is_anomaly'] = 1
data_pointanoms.loc[anomaly_dates, 'is_anomaly'] = 1

# ----------------------------------------------------------------------

#| label: collective anomaly creation
#| fig-cap: "17.5: Sales with collective anomalies. The circles mark a three-day sequence outlier which are extreme globally; crosses mark a three-day sequence anomaly that sits within normal global bounds but contextually anomalous."
#| fig-cap-location: bottom
#| fig-alt: "Sales line with two three-day regions highlighted: one sequence of circles well above local level, and one sequence of crosses elevated above local level."
#| message: false
#| echo: true
#| eval: true

data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data.set_index('date', inplace=True)
data = data.sort_index()

# Create point outlier and anomaly based on rolling std
point_outlier = pd.Timestamp('2022-06-04')
rolling_std_dev = data['sales'].rolling(window=14).std()

for i in range(3):
    d = point_outlier + pd.Timedelta(days=i)
    std_dev_rolling = rolling_std_dev.loc[d - pd.Timedelta(days=1)]
    data.loc[d, 'sales'] = (
        data.loc[d - pd.Timedelta(days=1), 'sales'] + ((5 - 1.5*i) * std_dev_rolling)
    )

point_anomaly = pd.Timestamp('2021-06-03')
for i in range(3):
    d = point_anomaly + pd.Timedelta(days=i)
    std_dev_rolling = rolling_std_dev.loc[d - pd.Timedelta(days=1)]
    data.loc[d, 'sales'] = (
    data.loc[d - pd.Timedelta(days=1), 'sales'] + ((1.1 - 0.2*i) * std_dev_rolling)   
)

# Rerun plot
fig, ax = plt.subplots()
ax.plot(data.index, data['sales'], color=custom_palette[0], label='Sales')

ax.scatter(
    [point_outlier + pd.Timedelta(days=i) for i in range(3)],
    [data.loc[point_outlier + pd.Timedelta(days=i), 'sales'] for i in range(3)],
    color=custom_palette[2], marker='o', s=100, label='Squence outliers', zorder=5
)
ax.scatter(
    [point_anomaly + pd.Timedelta(days=i) for i in range(3)],
    [data.loc[point_anomaly + pd.Timedelta(days=i), 'sales'] for i in range(3)],
    color=custom_palette[2], marker='x', s=100, label='Sequence anomaly', zorder=5
)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: save sequence anomaly labels
#| message: false
#| echo: false
#| eval: true

# save sequence outliers
seq_outlier_date = pd.Timestamp('2022-06-04')
seq_anomaly_date = pd.Timestamp('2021-06-03')

data_sequence = data.copy()
data_sequence['is_anomaly'] = 0

outlier_dates = pd.date_range(seq_outlier_date, periods=3)
anomaly_dates = pd.date_range(seq_anomaly_date, periods=3)

data_sequence.loc[outlier_dates, 'is_anomaly'] = 1
data_sequence.loc[anomaly_dates, 'is_anomaly'] = 1

# ----------------------------------------------------------------------

#| label: rules based detection
#| fig-cap: "17.7: Rules-based anomaly detection with floor and ceiling"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

stockouts_data = data_pointanoms.copy()

# Define stock-out dates
single_day_stockouts = [
    '2021-03-15', 
    '2021-07-04', 
    '2022-01-12', 
    '2022-11-20'
]

stockout_sequences = [
    pd.date_range(start='2021-05-10', end='2021-05-17'),  # 1 week outage
    pd.date_range(start='2022-08-01', end='2022-08-14'),  # 2 week outage
    pd.date_range(start='2022-12-24', end='2022-12-26')   # Holiday logistics failure
]

all_stockout_dates = pd.DatetimeIndex(single_day_stockouts).append(
    [period for period in stockout_sequences]
)

stockouts_data.loc[stockouts_data.index.isin(all_stockout_dates), 'sales'] = 0.0
stockouts_data.loc[stockouts_data['sales'] == 0, 'is_anomaly'] = 1

# Rules-based thresholds
floor_threshold = 5
ceiling_threshold = 270

total_anomalies = stockouts_data['is_anomaly'].sum()
n_above = (stockouts_data['sales'] > ceiling_threshold).sum()
n_below = (stockouts_data['sales'] < floor_threshold).sum()

# Original point anomaly dates (from data_pointanoms)
point_outlier_date = pd.Timestamp('2022-10-04')
point_anomaly_date = pd.Timestamp('2021-08-27')

fig, ax = plt.subplots()
ax.plot(stockouts_data.index, stockouts_data['sales'], color=custom_palette[0], label='Sales')
ax.axhline(y=floor_threshold, color=custom_palette[2], linestyle='--', linewidth=2, 
           label=f'Stock-out rule (x < {floor_threshold})')
ax.axhline(y=ceiling_threshold, color=custom_palette[2], linestyle='--', linewidth=2, 
           label=f'Stock-dump rule (x > {ceiling_threshold})')
ax.scatter(
    point_outlier_date,
    stockouts_data.loc[point_outlier_date, 'sales'],
    color=custom_palette[2], marker='o', s=100, label='Point Outlier', zorder=5
)
ax.scatter(
    point_anomaly_date,
    stockouts_data.loc[point_anomaly_date, 'sales'],
    color=custom_palette[2], marker='x', s=100, label='Point Anomaly', zorder=5
)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title(f'Rules based — Detected above: {n_above}, below: {n_below}, out of {total_anomalies}')
ax.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: x-chart static detection
#| fig-cap: "17.8: X-Chart static threshold detection"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

mean = stockouts_data['sales'].mean()
std = stockouts_data['sales'].std()

upper_limit = mean + 3 * std
lower_limit = mean - 3 * std

total_anomalies = int(stockouts_data['is_anomaly'].sum())
n_above = (stockouts_data['sales'] > upper_limit).sum()
n_below = (stockouts_data['sales'] < lower_limit).sum()
detected_above = stockouts_data[stockouts_data['sales'] > upper_limit]
detected_below = stockouts_data[stockouts_data['sales'] < lower_limit]

fig, ax = plt.subplots()
ax.plot(stockouts_data.index, stockouts_data['sales'], color='black', linewidth=2, label='Sales')

if not detected_above.empty:
    ax.scatter(detected_above.index, detected_above['sales'], 
               color=custom_palette[2], marker='o', s=100, 
               edgecolor='black', linewidth=1, zorder=5, 
               label=f'Detected Above ({len(detected_above)})')

if not detected_below.empty:
    ax.scatter(detected_below.index, detected_below['sales'], 
               color=custom_palette[3], marker='x', s=100, 
               linewidth=2, zorder=5, 
               label=f'Detected Below ({len(detected_below)})')
ax.scatter(
    point_anomaly_date,
    stockouts_data.loc[point_anomaly_date, 'sales'],
    color=custom_palette[2], marker='x', s=100, label='Point Anomaly', zorder=5
)

ax.axhline(y=mean, color=custom_palette[5], linestyle=':', linewidth=4, 
           label=f'Mean (μ = {mean:.2f})')
ax.axhline(y=upper_limit, color=custom_palette[2], linestyle='--', linewidth=2, 
           label=f'Upper (μ + 3σ = {upper_limit:.0f})')
ax.axhline(y=lower_limit, color=custom_palette[2], linestyle='--', linewidth=2, 
           label=f'Lower (μ - 3σ = {lower_limit:.0f})')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title(f'X-Chart (μ ± 3σ) — Detected above: {n_above}, below: {n_below}, out of {total_anomalies}')
ax.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: adaptive threshold detection
#| fig-cap: "17.10: Adaptive rolling threshold detection"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

# Adaptive thresholds: Rolling μ ± 3σ
window = 21  # 3 weeks

# Calculate rolling statistics (shift to avoid lookahead bias)
stockouts_data['rolling_mu'] = stockouts_data['sales'].rolling(window).mean().shift(1)
stockouts_data['rolling_sigma'] = stockouts_data['sales'].rolling(window).std().shift(1)

# Dynamic thresholds
stockouts_data['upper_limit'] = stockouts_data['rolling_mu'] + 3 * stockouts_data['rolling_sigma']
stockouts_data['lower_limit'] = stockouts_data['rolling_mu'] - 1.5 * stockouts_data['rolling_sigma']

# Detect anomalies
stockouts_data['detected'] = (
    (stockouts_data['sales'] > stockouts_data['upper_limit']) | 
    (stockouts_data['sales'] < stockouts_data['lower_limit'])
)

# Check which detections match actual anomalies
true_positives = (stockouts_data['detected'] & (stockouts_data['is_anomaly'] == 1)).sum()
false_positives = (stockouts_data['detected'] & (stockouts_data['is_anomaly'] == 0)).sum()
false_negatives = (~stockouts_data['detected'] & (stockouts_data['is_anomaly'] == 1)).sum()

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

# Which dates were detected vs missed?
detected_dates = stockouts_data[stockouts_data['detected']].index
known_anomaly_dates = stockouts_data[stockouts_data['is_anomaly'] == 1].index

correctly_detected = detected_dates.intersection(known_anomaly_dates)
missed = known_anomaly_dates.difference(detected_dates)
false_alarms = detected_dates.difference(known_anomaly_dates)

# ----------------------------------------------------------------------

#| label: adaptive threshold plot
#| fig-cap: "17.11: Adaptive threshold rolling mean with bounds"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(stockouts_data.index, stockouts_data['sales'], 
    color=custom_palette[0], alpha=0.7, linewidth=0.8, label='Sales')
ax.plot(stockouts_data.index, stockouts_data['rolling_mu'], 
    color=custom_palette[1], linewidth=1, alpha=0.7, label=f'Rolling Mean ({window}d)')
ax.fill_between(stockouts_data.index, 
        stockouts_data['lower_limit'], 
        stockouts_data['upper_limit'],
        color=custom_palette[1], alpha=0.2, label='±3σ bounds')

# Mark detections
detected = stockouts_data[stockouts_data['detected'] == True]
tp = detected[detected['is_anomaly'] == 1]
fp = detected[detected['is_anomaly'] == 0]

ax.scatter(tp.index, tp['sales'], color=custom_palette[2], s=40, zorder=5,
       marker='o', label=f'True Positive ({len(tp)})')
if len(fp) > 0:
    ax.scatter(fp.index, fp['sales'], color=custom_palette[4], s=40, zorder=5,
           marker='x', label=f'False Positive ({len(fp)})')

ax.set_ylabel('Sales')
ax.set_xlabel('Date')
ax.set_title(f'Adaptive Thresholds: Rolling μ +3σ upper / −1.5σ (window={window} days)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: conditional x-chart detection
#| fig-cap: "17.12: Conditional X-chart with regime-based thresholds"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

alpha = 0.4 # 40% of mean (regular sales condition)

def get_regime(row): 
    if row['price'] == 5.0: 
        return 'regular' 
    elif row['price'] == 3.5: 
        return 'promo_low' 
    else: # price == 2.5 
        return 'promo_high'

stockouts_data['regime'] = stockouts_data.apply(get_regime, axis=1)

# σ upper bound per regime
regime_stats = stockouts_data.groupby('regime')['sales'].agg(['mean', 'std'])
regime_stats['upper'] = regime_stats['mean'] + 3 * regime_stats['std']
stockouts_data['upper'] = stockouts_data['regime'].map(regime_stats['upper'])

# Regular-anchored proportional floor 
mu_regular = stockouts_data.loc[stockouts_data['regime'] == 'regular', 'sales'].mean()
stockouts_data['prop_lower'] = alpha * mu_regular

# Bounds using proportional floor and std based upper bound based on regime
stockouts_data['detected'] = (
    (stockouts_data['sales'] < stockouts_data['prop_lower']) |
    (stockouts_data['sales'] > stockouts_data['upper'])   
)


# Confusion vs ground-truth is_anomaly (expects 0/1)
is_true = stockouts_data['is_anomaly'].fillna(0).astype(int) == 1
is_det  = stockouts_data['detected'].astype(bool)

tp = (is_det & is_true).sum()
fp = (is_det & ~is_true).sum()
fn = (~is_det & is_true).sum()

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0

# values for inspection
detected_dates = stockouts_data.index[is_det]
known_anomaly_dates = stockouts_data.index[is_true]
correctly_detected = detected_dates.intersection(known_anomaly_dates)
missed = known_anomaly_dates.difference(detected_dates)
false_alarms = detected_dates.difference(known_anomaly_dates)

# ----------------------------------------------------------------------

#| label: conditional x-chart plot
#| fig-cap: "17.13: Conditional X-chart detection results"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true
fig, ax = plt.subplots(figsize=(16, 6))

# Plot sales colored by regime
colors = {'regular': custom_palette[0], 'promo_low': custom_palette[3], 'promo_high': custom_palette[1]}
for regime in ['regular', 'promo_low', 'promo_high']:
    regime_data = stockouts_data[stockouts_data['regime'] == regime]
    ax.scatter(regime_data.index, regime_data['sales'],
               color=colors[regime], s=12, label=f'{regime}')

# upper bound 
if 'upper' in stockouts_data.columns:
    ax.step(stockouts_data.index, stockouts_data['upper'],
            color=custom_palette[2], linestyle='--', linewidth=2, where='mid', label='Upper bound (μ+3σ)')

# proportional lower bound anchored to regular mean
ax.step(stockouts_data.index, stockouts_data['prop_lower'],
        color=custom_palette[4], linestyle='-', linewidth=2, where='mid',
        label=rf'Primary floor: ($\alpha={alpha}$)')

# Mark detections vs truth: TP / FP
detected = stockouts_data[stockouts_data['detected']]
tp_df = detected[detected['is_anomaly'].fillna(0).astype(int) == 1]
fp_df = detected[detected['is_anomaly'].fillna(0).astype(int) == 0]

ax.scatter(tp_df.index, tp_df['sales'], color=custom_palette[2], s=60, zorder=5,
           marker='o', edgecolor='black', linewidth=1, label=f'TP ({len(tp_df)})')

if len(fp_df) > 0:
    ax.scatter(fp_df.index, fp_df['sales'], color=custom_palette[4], s=60, zorder=5,
               marker='x', linewidth=2, label=f'FP ({len(fp_df)})')

ax.set_ylabel('Sales')
ax.set_xlabel('Date')
ax.set_title(f'Conditional X-Chart | Precision={precision:.2f}, Recall={recall:.2f}')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: contaminated profile model
#| message: false
#| echo: true
#| eval: true

## Contaminated profile
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Prepare statsforecast long format
sf_data = stockouts_data[['day_of_week', 'week_num', 'promotion', 'price', 'sales', 'is_holiday', 'is_anomaly']].copy()
sf_data = sf_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
sf_data['unique_id'] = 'product_1'

# Exogenous columns
exog_cols = ['day_of_week', 'week_num', 'promotion', 'price', 'is_holiday']

# Training df: unique_id + ds + y + exogenous columns together
train_cols = ['unique_id', 'ds', 'y'] + exog_cols

# Future exogenous — one row for the throwaway h=1 forecast
last_row = sf_data.iloc[-1]
last_date = sf_data['ds'].max()
X_future = pd.DataFrame({
    'unique_id': ['product_1'],
    'ds': [last_date + pd.Timedelta(days=1)],
    'day_of_week': [(last_date + pd.Timedelta(days=1)).dayofweek],
    'week_num': [last_row['week_num']],
    'promotion': [last_row['promotion']],
    'price': [last_row['price']],
    'is_holiday': [False]
})

sf_contaminated = StatsForecast(
    models=[AutoARIMA()],
    freq='D',
    n_jobs=1
)

# Exogenous go IN df, X_df is future-only
sf_contaminated.forecast(
    df=sf_data[train_cols],
    X_df=X_future,
    h=1,
    fitted=True
)

fitted_c = sf_contaminated.forecast_fitted_values()

# Reset index if fitted_c has unique_id/ds as index
fitted_c = fitted_c.reset_index()

sf_data = sf_data.merge(
    fitted_c[['unique_id', 'ds', 'AutoARIMA']],
    on=['unique_id', 'ds'],
    how='left'
)

stockouts_data['y_hat_contaminated'] = sf_data['AutoARIMA'].values
stockouts_data['resid_contaminated'] = stockouts_data['sales'] - stockouts_data['y_hat_contaminated']

# ----------------------------------------------------------------------

#| label: z-score contaminated detection
#| message: false
#| echo: true
#| eval: true

# Z-score detection on contaminated residuals
mu_c  = stockouts_data['resid_contaminated'].mean()
sig_c = stockouts_data['resid_contaminated'].std()

stockouts_data['z_contaminated'] = (
    (stockouts_data['resid_contaminated'] - mu_c) / sig_c
)

threshold = 3.0
stockouts_data['detected_contaminated'] = stockouts_data['z_contaminated'].abs() > threshold

# Evaluate against ground truth
is_true = stockouts_data['is_anomaly'].fillna(0).astype(int) == 1
is_det  = stockouts_data['detected_contaminated'].astype(bool)

tp_c = (is_det &  is_true).sum()
fp_c = (is_det & ~is_true).sum()
fn_c = (~is_det & is_true).sum()

precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
recall_c    = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
print(f"Contaminated — Precision: {precision_c:.2f}  Recall: {recall_c:.2f}")
print(f"  TP: {tp_c}  FP: {fp_c}  FN: {fn_c}")

# ----------------------------------------------------------------------

#| label: contaminated profile plot
#| fig-cap: "17.14: Contaminated profile model with Z-score detection"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True)

# Panel 1 - sales vs contaminated profile
axes[0].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.9, label='Sales')
axes[0].plot(stockouts_data.index, stockouts_data['y_hat_contaminated'],
             color=custom_palette[2], linewidth=1.2,
             linestyle='--', label='Profile (contaminated)')
axes[0].set_ylabel('Sales')
axes[0].legend()

# Panel 2 - residual Z-score
axes[1].axhline( threshold, color=custom_palette[2], linestyle='--', linewidth=1.5)
axes[1].axhline(-threshold, color=custom_palette[2], linestyle='--', linewidth=1.5)
axes[1].axhline(0, color='grey', linestyle=':', linewidth=1)
axes[1].plot(stockouts_data.index, stockouts_data['z_contaminated'],
             color=custom_palette[0], linewidth=0.8, label='Residual Z-score')

tp_df = stockouts_data[is_det &  is_true]
fp_df = stockouts_data[is_det & ~is_true]

axes[1].scatter(tp_df.index, tp_df['z_contaminated'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_df)})')
if len(fp_df) > 0:
    axes[1].scatter(fp_df.index, fp_df['z_contaminated'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_df)})')

axes[1].set_ylabel('Z-score')
axes[1].set_xlabel('Date')
axes[1].legend()
fig.suptitle(
    f'Contaminated Profile | Precision={precision_c:.2f}, Recall={recall_c:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: derive-residual-signal
#| message: false

# rename cols
stockouts_data['y_hat']            = stockouts_data.pop('y_hat_contaminated')
stockouts_data['residual']         = stockouts_data.pop('resid_contaminated')
stockouts_data['residual_z']       = stockouts_data.pop('z_contaminated')
stockouts_data['detected_profile'] = stockouts_data.pop('detected_contaminated')

# ----------------------------------------------------------------------

#| label: pyod import
#| message: false
#| echo: true
#| eval: true

# pip install pyod
from pyod.models.iforest import IForest

# ----------------------------------------------------------------------

#| label: iforest unsupervised tuning
#| message: false
#| echo: false
#| eval: false

from pyod.models.iforest import IForest
from itertools import product

X_if_v0 = stockouts_data[['day_of_week', 'week_num', 'promotion', 'price', 'sales', 'is_holiday']]

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_samples': [0.5, 0.8, 1.0],
    'max_features': [0.5, 0.75, 1.0],
    'contamination': [0.03, 0.05, 0.08]
}

results = []
keys = list(param_grid.keys())
for vals in product(*param_grid.values()):
    params = dict(zip(keys, vals))
    model = IForest(random_state=456, **params)
    model.fit(X_if_v0)
    preds = model.predict(X_if_v0) == 1
    
    tp_t = (preds & is_true).sum()
    fp_t = (preds & ~is_true).sum()
    fn_t = (~preds & is_true).sum()
    
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) else 0.0
    
    results.append({**params, 'precision': prec_t, 'recall': rec_t, 'f1': f1_t})

results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
results_df.head(10)

# ----------------------------------------------------------------------

#| label: iforest unsupervised fit
#| message: false
#| echo: true
#| eval: true

from pyod.models.iforest import IForest

X_if_v0 = stockouts_data[['day_of_week', 'week_num', 'promotion', 'price', 'sales', 'is_holiday']]

iso_full = IForest(
    n_estimators=500,
    max_samples=0.5,
    max_features=0.5,
    contamination=0.08,
    random_state=456
)
iso_full.fit(X_if_v0)

stockouts_data['if_unsup_score'] = iso_full.decision_function(X_if_v0)
stockouts_data['if_unsup_label'] = iso_full.predict(X_if_v0)
stockouts_data['detected_if_unsup'] = stockouts_data['if_unsup_label'] == 1

is_det_iu = stockouts_data['detected_if_unsup'].astype(bool)
tp_iu = (is_det_iu &  is_true).sum()
fp_iu = (is_det_iu & ~is_true).sum()
fn_iu = (~is_det_iu & is_true).sum()

prec_iu = tp_iu / (tp_iu + fp_iu) if (tp_iu + fp_iu) else 0.0
rec_iu  = tp_iu / (tp_iu + fn_iu) if (tp_iu + fn_iu) else 0.0
print(f"IF (fully unsupervised) — Precision: {prec_iu:.2f}  Recall: {rec_iu:.2f}")

# ----------------------------------------------------------------------

#| label: iforest unsupervised plot
#| fig-cap: "17.15: Isolation Forest unsupervised detection"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(stockouts_data.index, stockouts_data['if_unsup_score'],
             color=custom_palette[1], linewidth=0.9, label='IF Anomaly Score')
axes[0].set_ylabel('Anomaly Score')
axes[0].legend()

axes[1].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.8, label='Sales')

tp_iu_df = stockouts_data[is_det_iu &  is_true]
fp_iu_df = stockouts_data[is_det_iu & ~is_true]

axes[1].scatter(tp_iu_df.index, tp_iu_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_iu_df)})')
if len(fp_iu_df) > 0:
    axes[1].scatter(fp_iu_df.index, fp_iu_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_iu_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend()
fig.suptitle(
    f'IF Unsupervised (Raw Features) | Precision={prec_iu:.2f}, Recall={rec_iu:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: iforest residuals tuning
#| message: false
#| echo: false
#| eval: false

X_if_v1 = stockouts_data[['residual', 'residual_z', 'price']].fillna(0)

results_resid = []
for vals in product(*param_grid.values()):
    params = dict(zip(keys, vals))
    model = IForest(random_state=456, **params)
    model.fit(X_if_v1)
    preds = model.predict(X_if_v1) == 1
    
    tp_t = (preds & is_true).sum()
    fp_t = (preds & ~is_true).sum()
    fn_t = (~preds & is_true).sum()
    
    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) else 0.0
    
    results_resid.append({**params, 'precision': prec_t, 'recall': rec_t, 'f1': f1_t})

results_resid_df = pd.DataFrame(results_resid).sort_values('f1', ascending=False)
best_resid = results_resid_df.iloc[0]
print(f"Best IF on residuals — Precision: {best_resid['precision']:.2f}  Recall: {best_resid['recall']:.2f}")
results_resid_df.head(5)

# ----------------------------------------------------------------------

#| label: iforest residuals 
#| message: false
#| echo: true
#| eval: true

X_if_v1 = stockouts_data[['residual', 'residual_z', 'price']].fillna(0)

# Best params from tuning above
iso_resid = IForest(
    n_estimators=500,
    max_samples=0.8,
    max_features=0.5,
    contamination=0.08,
    random_state=456
)
iso_resid.fit(X_if_v1)

stockouts_data['if_score'] = iso_resid.decision_function(X_if_v1)
stockouts_data['if_label'] = iso_resid.predict(X_if_v1)
stockouts_data['detected_if'] = stockouts_data['if_label'] == 1

is_det_if = stockouts_data['detected_if'].astype(bool)
tp_if = (is_det_if &  is_true).sum()
fp_if = (is_det_if & ~is_true).sum()
fn_if = (~is_det_if & is_true).sum()

prec_if = tp_if / (tp_if + fp_if) if (tp_if + fp_if) else 0.0
rec_if  = tp_if / (tp_if + fn_if) if (tp_if + fn_if) else 0.0
print(f"IF on residuals — Precision: {prec_if:.2f}  Recall: {rec_if:.2f}")

# ----------------------------------------------------------------------

#| label: iforest residuals plot
#| fig-cap: "17.16: Isolation Forest on residuals"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(stockouts_data.index, stockouts_data['if_score'],
             color=custom_palette[1], linewidth=0.9, label='IF Anomaly Score')
axes[0].set_ylabel('Anomaly Score')
axes[0].legend()

axes[1].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.8, label='Sales')

tp_if_df = stockouts_data[is_det_if &  is_true]
fp_if_df = stockouts_data[is_det_if & ~is_true]

axes[1].scatter(tp_if_df.index, tp_if_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_if_df)})')
if len(fp_if_df) > 0:
    axes[1].scatter(fp_if_df.index, fp_if_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_if_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend()
fig.suptitle(
    f'IF on Residuals | Precision={prec_if:.2f}, Recall={rec_if:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: inne extended iforest
#| message: false
#| echo: true
#| eval: true

from pyod.models.inne import INNE

X_if_v0 = stockouts_data[['day_of_week', 'week_num', 'promotion', 'price', 'sales', 'is_holiday']].copy()

inne_model = INNE(
    n_estimators=500,
    max_samples=0.5,
    contamination=0.08,
    random_state=456
)
inne_model.fit(X_if_v0)

stockouts_data['eif_score'] = inne_model.decision_function(X_if_v0)
stockouts_data['eif_label'] = inne_model.predict(X_if_v0)
stockouts_data['detected_eif'] = stockouts_data['eif_label'] == 1

is_det_eif = stockouts_data['detected_eif'].astype(bool)
tp_eif = (is_det_eif &  is_true).sum()
fp_eif = (is_det_eif & ~is_true).sum()
fn_eif = (~is_det_eif & is_true).sum()

prec_eif = tp_eif / (tp_eif + fp_eif) if (tp_eif + fp_eif) else 0.0
rec_eif  = tp_eif / (tp_eif + fn_eif) if (tp_eif + fn_eif) else 0.0
print(f"INNE (Extended IF) — Precision: {prec_eif:.2f}  Recall: {rec_eif:.2f}")

# ----------------------------------------------------------------------

#| label: lof detection
#| message: false
#| echo: true
#| eval: true

from pyod.models.lof import LOF

lof_model = LOF(
    n_neighbors=50,
    contamination=0.05,
    algorithm='auto'
)
lof_model.fit(X_if_v0)

stockouts_data['lof_score'] = lof_model.decision_function(X_if_v0)
stockouts_data['lof_label'] = lof_model.predict(X_if_v0)
stockouts_data['detected_lof'] = stockouts_data['lof_label'] == 1

is_det_lof = stockouts_data['detected_lof'].astype(bool)
tp_lof = (is_det_lof &  is_true).sum()
fp_lof = (is_det_lof & ~is_true).sum()
fn_lof = (~is_det_lof & is_true).sum()

prec_lof = tp_lof / (tp_lof + fp_lof) if (tp_lof + fp_lof) else 0.0
rec_lof  = tp_lof / (tp_lof + fn_lof) if (tp_lof + fn_lof) else 0.0
print(f"LOF — Precision: {prec_lof:.2f}  Recall: {rec_lof:.2f}")

# ----------------------------------------------------------------------

#| label: lof detection plot
#| fig-cap: "17.17: Local Outlier Factor detection"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(stockouts_data.index, stockouts_data['lof_score'],
             color=custom_palette[1], linewidth=0.9, label='LOF Score')
axes[0].set_ylabel('Anomaly Score')
axes[0].legend()

axes[1].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.8, label='Sales')

tp_lof_df = stockouts_data[is_det_lof &  is_true]
fp_lof_df = stockouts_data[is_det_lof & ~is_true]

axes[1].scatter(tp_lof_df.index, tp_lof_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_lof_df)})')
if len(fp_lof_df) > 0:
    axes[1].scatter(fp_lof_df.index, fp_lof_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_lof_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend()
fig.suptitle(
    f'LOF (k={lof_model.n_neighbors}) | Precision={prec_lof:.2f}, Recall={rec_lof:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: unsupervised comparison table
#| tbl-cap: "17.18: Unsupervised method comparison"
#| message: false
#| echo: false
#| eval: false


unsup_results = pd.DataFrame({
    'Method': [
        'Z-score Profile (contaminated)',
        'IF — Raw Features (unsupervised)',
        'IF — Residuals',
        'INNE (Extended IF) — Residuals',
        'LOF — Residuals'
    ],
    'Features': [
        'resid_contaminated',
        'sales + context (no model)',
        'residual, residual_z, price',
        'residual, residual_z, price',
        'residual, residual_z, price'
    ],
    'Precision': [precision, prec_iu, prec_if, prec_eif, prec_lof],
    'Recall':    [recall,    rec_iu,  rec_if,  rec_eif,  rec_lof],
})
unsup_results['F1'] = (
    2 * unsup_results['Precision'] * unsup_results['Recall'] 
    / (unsup_results['Precision'] + unsup_results['Recall'])
).round(2)#.replace([np.inf, -np.inf, np.nan], 0)
unsup_results['Precision'] = unsup_results['Precision'].round(2)
unsup_results['Recall'] = unsup_results['Recall'].round(2)

print(unsup_results.to_markdown(index=False))

# ----------------------------------------------------------------------

#| label: mp univariate compute
#| message: false
#| echo: true
#| eval: true

import stumpy

# Matrix Profile on raw sales — no profile model needed
sales = stockouts_data['sales'].values.astype(float)
m = 28  # four weekly cycles

mp = stumpy.stump(sales, m=m, normalize=False) 
profile_values = mp[:, 0].astype(float)

# Align back to full index
mp_index = stockouts_data.index[:len(profile_values)]
stockouts_data['mp_score'] = np.nan
stockouts_data.loc[mp_index, 'mp_score'] = profile_values

mp_threshold = np.nanpercentile(profile_values, 95)
stockouts_data['detected_mp'] = stockouts_data['mp_score'] > mp_threshold

is_det_mp = stockouts_data['detected_mp'].fillna(False).astype(bool)
tp_mp = (is_det_mp &  is_true).sum()
fp_mp = (is_det_mp & ~is_true).sum()
fn_mp = (~is_det_mp & is_true).sum()

prec_mp = tp_mp / (tp_mp + fp_mp) if (tp_mp + fp_mp) else 0.0
rec_mp  = tp_mp / (tp_mp + fn_mp) if (tp_mp + fn_mp) else 0.0
print(f"Matrix Profile (univariate) — Precision: {prec_mp:.2f}  Recall: {rec_mp:.2f}")

# ----------------------------------------------------------------------

#| label: mp univariate plot
#| fig-cap: "17.19: Univariate Matrix Profile discord detection"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True,
                          figsize=(CFG.img_dim1, CFG.img_dim2))

axes[0].plot(stockouts_data.index, stockouts_data['mp_score'],
             color=custom_palette[1], linewidth=0.8, label='MP Score')
axes[0].axhline(mp_threshold, color=custom_palette[2],
                linestyle='--', linewidth=1.5,
                label=f'Threshold (95th pct)')
axes[0].set_ylabel('MP Score')
axes[0].legend(loc='upper left')

axes[1].plot(stockouts_data.index, stockouts_data['sales'], color=custom_palette[0], linewidth=0.8, label='Sales')

tp_mp_df = stockouts_data[is_det_mp &  is_true]
fp_mp_df = stockouts_data[is_det_mp & ~is_true]

axes[1].scatter(tp_mp_df.index, tp_mp_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_mp_df)})')
if len(fp_mp_df) > 0:
    axes[1].scatter(fp_mp_df.index, fp_mp_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_mp_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper left')

fig.suptitle(
    f'Matrix Profile (Univariate, Sales) | m={m} | '
    f'Precision={prec_mp:.2f}, Recall={rec_mp:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: mp multidimensional compute
#| message: false
#| echo: true
#| eval: true

# Build multi-dimensional input, with each row = one dimension
# Sales is still target; but we add price for exogenous context
T_multi = np.array([
    stockouts_data['sales'].values.astype(float),
    stockouts_data['price'].values.astype(float),
])

m = 28

# Appling include=[0] forces sales (row 0) to always be in the subspace
# discords=True reverses distances so profile favours large values
mps, indices = stumpy.mstump(T_multi, m=m, include=[0], discords=True, normalize=True)

# Using a 2-dimensional profile (row index 1, i.e. k=2: both sales and price)
# This profile considers both dimensions jointly
profile_multi = mps[1]  # k=2 dimensional profile

mp_index = stockouts_data.index[:len(profile_multi)]
stockouts_data['mp_multi_score'] = np.nan
stockouts_data.loc[mp_index, 'mp_multi_score'] = profile_multi

mp_multi_threshold = np.nanpercentile(profile_multi, 95)
stockouts_data['detected_mp_multi'] = (stockouts_data['mp_multi_score'] > mp_multi_threshold)

is_det_mpm = stockouts_data['detected_mp_multi'].fillna(False).astype(bool)
tp_mpm = (is_det_mpm &  is_true).sum()
fp_mpm = (is_det_mpm & ~is_true).sum()
fn_mpm = (~is_det_mpm & is_true).sum()

prec_mpm = tp_mpm / (tp_mpm + fp_mpm) if (tp_mpm + fp_mpm) else 0.0
rec_mpm  = tp_mpm / (tp_mpm + fn_mpm) if (tp_mpm + fn_mpm) else 0.0
print(f"Matrix Profile (multi, sales+price) — Precision: {prec_mpm:.2f}  Recall: {rec_mpm:.2f}")

# ----------------------------------------------------------------------

#| label: mp multidimensional plot
#| fig-cap: "17.20: Multidimensional Matrix Profile (sales + price)"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(2, 1, sharex=True,
                          figsize=(CFG.img_dim1, CFG.img_dim2))

axes[0].plot(stockouts_data.index, stockouts_data['mp_multi_score'],
             color=custom_palette[1], linewidth=0.8, label='MP Score (multi)')
axes[0].axhline(mp_multi_threshold, color=custom_palette[2],
                linestyle='--', linewidth=1.5,
                label=f'Threshold (95th pct)')
axes[0].set_ylabel('MP Score')
axes[0].legend(loc='upper left')

axes[1].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.8, label='Sales')

tp_mpm_df = stockouts_data[is_det_mpm &  is_true]
fp_mpm_df = stockouts_data[is_det_mpm & ~is_true]

axes[1].scatter(tp_mpm_df.index, tp_mpm_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_mpm_df)})')
if len(fp_mpm_df) > 0:
    axes[1].scatter(fp_mpm_df.index, fp_mpm_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_mpm_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper left')

fig.suptitle(
    f'Matrix Profile (Multidimensional, Sales+Price) | m={m} | '
    f'Precision={prec_mpm:.2f}, Recall={rec_mpm:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: mp window length comparison
#| fig-cap: "17.21: Matrix Profile score sensitivity to window length"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

# Compare profile scores across window lengths

sales = stockouts_data['sales'].values.astype(float)

fig, axes = plt.subplots(3, 1, sharex=True,
                          figsize=(CFG.img_dim1, CFG.img_dim2 * 1.4))

for ax, m_val in zip(axes, [3, 7, 14]):
    mp_val = stumpy.stump(sales, m=m_val, normalize=False)
    pv = mp_val[:, 0].astype(float)
    idx = stockouts_data.index[:len(pv)]

    ax.plot(idx, pv, color=custom_palette[1],
            linewidth=0.8, label=f'm={m_val}')
    ax.axhline(np.percentile(pv, 95),
               color=custom_palette[2], linestyle='--',
               linewidth=1, alpha=0.7)
    ax.set_ylabel('MP Score')
    ax.legend(loc='upper left')

axes[-1].set_xlabel('Date')
fig.suptitle('Matrix Profile Score — Sensitivity to Window Length m')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: pan matrix profile
#| fig-cap: "17.22: Multi-scale Matrix Profile consensus across window lengths"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

sales = stockouts_data['sales'].values.astype(float)
window_sizes = [7, 14, 21, 28]

# Compute profiles at each scale and build consensus
raw_profiles = np.full((len(window_sizes), len(sales)), np.nan)

for i, m_val in enumerate(window_sizes):
    mp_val = stumpy.stump(sales, m=m_val, normalize=False)
    pv = mp_val[:, 0].astype(float)
    pv[np.isinf(pv)] = np.nan
    # Min-max normalise each profile to make scales comparable
    pv_norm = (pv - np.nanmin(pv)) / (np.nanmax(pv) - np.nanmin(pv))
    raw_profiles[i, :len(pv_norm)] = pv_norm

consensus = np.nanmax(raw_profiles, axis=0)
idx = stockouts_data.index[:len(consensus)]

fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
ax.plot(idx, consensus, color=custom_palette[1], linewidth=0.8, 
        label='Consensus (max across scales)')
ax.axhline(np.nanpercentile(consensus, 95), color=custom_palette[2],
           linestyle='--', linewidth=1.5, label='95th percentile')
ax.set_xlabel('Date')
ax.set_ylabel('Normalised MP Score')
ax.set_title('Multi-Scale Matrix Profile Consensus')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: mp ab join
#| message: false
#| echo: true
#| eval: true

# Clean reference data
clean_mask = (stockouts_data.index >= "2021-01-13") & (stockouts_data.index <= "2022-07-31")
sales_clean = stockouts_data.loc[clean_mask, "sales"].values.astype(float)

mp_ab = stumpy.stump(T_A=sales, m=m, T_B=sales_clean, ignore_trivial=False)
profile_ab = mp_ab[:, 0].astype(float)

mp_nn_index = stockouts_data.index[:len(profile_ab)]  # <-- alignment index

stockouts_data["mp_ab_score"] = np.nan
stockouts_data.loc[mp_nn_index, "mp_ab_score"] = profile_ab

mp_ab_threshold = np.nanpercentile(profile_ab, 95)
stockouts_data['detected_mp_ab'] = stockouts_data['mp_ab_score'] > mp_ab_threshold

is_det_ab = stockouts_data['detected_mp_ab'].fillna(False).astype(bool)
tp_ab = (is_det_ab &  is_true).sum()
fp_ab = (is_det_ab & ~is_true).sum()
fn_ab = (~is_det_ab & is_true).sum()

prec_ab = tp_ab / (tp_ab + fp_ab) if (tp_ab + fp_ab) else 0.0
rec_ab  = tp_ab / (tp_ab + fn_ab) if (tp_ab + fn_ab) else 0.0
print(f"MP AB-join (sales vs clean ref) — Precision: {prec_ab:.2f}  Recall: {rec_ab:.2f}")

# ----------------------------------------------------------------------

#| label: mp window overlap evaluation
#| message: false
#| echo: true
#| eval: true

def window_overlap_eval(detected_starts, is_true, m, n):
    """
    Expand window-start detections to full [i, i+m) coverage,
    then compute point-level precision and recall against ground truth.
    """
    detected_expanded = np.zeros(n, dtype=bool)
    for i in np.where(detected_starts)[0]:
        detected_expanded[i:min(i + m, n)] = True
    
    tp = (detected_expanded &  is_true.values).sum()
    fp = (detected_expanded & ~is_true.values).sum()
    fn = (~detected_expanded & is_true.values).sum()
    
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec

n = len(stockouts_data)

# Re-evaluate univariate MP with window overlap
prec_mp_w, rec_mp_w = window_overlap_eval(
    is_det_mp.values, is_true, m, n
)
print(f"MP univariate (window-overlap) — Precision: {prec_mp_w:.2f}  Recall: {rec_mp_w:.2f}")

# Re-evaluate multidimensional MP with window overlap
prec_mpm_w, rec_mpm_w = window_overlap_eval(
    is_det_mpm.values, is_true, m, n
)
print(f"MP multi (window-overlap) — Precision: {prec_mpm_w:.2f}  Recall: {rec_mpm_w:.2f}")

# ----------------------------------------------------------------------

#| label: catboost supervised classifier
#| message: false
#| echo: true
#| eval: true

from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_recall_curve
import pandas as pd
import numpy as np

def build_supervised_features(df, window=14):
    """
    Feature matrix for supervised residual profiling.
    All rolling features are shifted by 1 to avoid lookahead bias.
    """
    feats = pd.DataFrame(index=df.index)

    # Core residual signal
    feats['residual']          = df['residual']
    feats['residual_z']        = df['residual_z']
    feats['residual_abs']      = df['residual'].abs()

    # Local residual statistics
    for w in [7, 14, 28]:
        feats[f'resid_roll_mean_{w}'] = (
            df['residual'].rolling(w, min_periods=3).mean().shift(1)
        )
        feats[f'resid_roll_std_{w}'] = (
            df['residual'].rolling(w, min_periods=3).std().shift(1)
        )
        feats[f'resid_roll_max_{w}'] = (
            df['residual'].abs().rolling(w, min_periods=3).max().shift(1)
        )

    # Scores from unsupervised detectors as features
    for col in ['if_unsup_score', 'if_score', 'eif_score', 'lof_score']:
        if col in df.columns:
            feats[col] = df[col]

    # Context
    feats['price']             = df['price']
    feats['promotion']         = df['promotion']
    feats['sales_roll_mean']   = (
        df['sales'].rolling(14, min_periods=3).mean().shift(1)
    )
    feats['day_of_week']       = df.index.dayofweek
    feats['month']             = df.index.month

    return feats.fillna(0)


feature_df = build_supervised_features(stockouts_data)
labels     = stockouts_data['is_anomaly'].fillna(0).astype(int)

# Time-series split
tscv = TimeSeriesSplit(n_splits=4)

all_preds  = np.zeros(len(labels))
all_probs  = np.zeros(len(labels))
test_mask  = np.zeros(len(labels), dtype=bool)

for train_idx, test_idx in tscv.split(feature_df):
    X_tr = feature_df.iloc[train_idx].values
    X_te = feature_df.iloc[test_idx].values
    y_tr = labels.iloc[train_idx].values
    y_te = labels.iloc[test_idx].values

    clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        auto_class_weights='Balanced',
        eval_metric='F1',
        random_seed=42,
        verbose=0
    )
    clf.fit(X_tr, y_tr, eval_set=(X_te, y_te))

    all_preds[test_idx]  = clf.predict(X_te)
    all_probs[test_idx]  = clf.predict_proba(X_te)[:, 1]
    test_mask[test_idx]  = True

y_test_all  = labels.values[test_mask]
y_pred_all  = all_preds[test_mask].astype(int)
y_prob_all  = all_probs[test_mask]

print(classification_report(y_test_all, y_pred_all,
                            target_names=['Normal', 'Anomaly']))

# ----------------------------------------------------------------------

#| label: catboost hyperparameter search
#| message: false
#| echo: false
#| eval: false

from itertools import product as iprod

cb_param_grid = {
    'iterations': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 6],
}

cb_results = []
for iters, lr, depth in iprod(
    cb_param_grid['iterations'],
    cb_param_grid['learning_rate'],
    cb_param_grid['depth']
):
    fold_f1s = []
    for train_idx, test_idx in tscv.split(feature_df):
        X_tr = feature_df.iloc[train_idx].values
        X_te = feature_df.iloc[test_idx].values
        y_tr = labels.iloc[train_idx].values
        y_te = labels.iloc[test_idx].values

        m = CatBoostClassifier(
            iterations=iters, learning_rate=lr, depth=depth,
            auto_class_weights='Balanced', eval_metric='F1',
            random_seed=42, verbose=0
        )
        m.fit(X_tr, y_tr, eval_set=(X_te, y_te))
        preds_te = m.predict(X_te).astype(int)
        
        tp_t = ((preds_te == 1) & (y_te == 1)).sum()
        fp_t = ((preds_te == 1) & (y_te == 0)).sum()
        fn_t = ((preds_te == 0) & (y_te == 1)).sum()
        
        p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
        r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        fold_f1s.append(f)
    
    cb_results.append({
        'iterations': iters, 'learning_rate': lr, 'depth': depth,
        'mean_f1': np.mean(fold_f1s), 'std_f1': np.std(fold_f1s)
    })

cb_results_df = pd.DataFrame(cb_results).sort_values('mean_f1', ascending=False)
cb_results_df.head(10)

# ----------------------------------------------------------------------

#| label: catboost pr curve and importance
#| fig-cap: "17.23: CatBoost precision-recall curve and feature importance"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

fig, axes = plt.subplots(1, 2, figsize=(CFG.img_dim1, CFG.img_dim2 * 0.6))

# Panel 1: precision-recall curve
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test_all, y_prob_all)
axes[0].plot(rec_curve, prec_curve, color=custom_palette[1], linewidth=2)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].axhline(y=labels.mean(), color='grey', linestyle='--', linewidth=1,
                label=f'Baseline ({labels.mean():.3f})')
axes[0].legend()

# Panel 2: feature importance (last fold)
importances = pd.Series(
    clf.get_feature_importance(), index=feature_df.columns
).sort_values(ascending=True)

importances.tail(10).plot(kind='barh', ax=axes[1], color=custom_palette[1])
axes[1].set_title('Feature Importance (CatBoost)')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: threshold adjustment
#| message: false
#| echo: true
#| eval: true

from sklearn.metrics import f1_score

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [
    f1_score(y_test_all, (y_prob_all >= t).astype(int), zero_division=0)
    for t in thresholds
]

best_thresh = thresholds[np.argmax(f1_scores)]
best_f1     = max(f1_scores)

y_pred_tuned = (y_prob_all >= best_thresh).astype(int)
tp_sup = ((y_pred_tuned == 1) & (y_test_all == 1)).sum()
fp_sup = ((y_pred_tuned == 1) & (y_test_all == 0)).sum()
fn_sup = ((y_pred_tuned == 0) & (y_test_all == 1)).sum()

prec_sup = tp_sup / (tp_sup + fp_sup) if (tp_sup + fp_sup) else 0.0
rec_sup  = tp_sup / (tp_sup + fn_sup) if (tp_sup + fn_sup) else 0.0

print(f"Best threshold: {best_thresh:.2f}")
print(f"Supervised (tuned): Precision: {prec_sup:.2f}  Recall: {rec_sup:.2f}  F1: {best_f1:.2f}")

# ----------------------------------------------------------------------

#| label: catboost detections plot
#| fig-cap: "17.24: CatBoost supervised detections with tuned threshold"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Map tuned predictions back to full index for plotting
stockouts_data['sup_prob'] = np.nan
stockouts_data.iloc[np.where(test_mask)[0], stockouts_data.columns.get_loc('sup_prob')] = y_prob_all
stockouts_data['detected_sup'] = stockouts_data['sup_prob'] >= best_thresh

is_det_sup = stockouts_data['detected_sup'].fillna(False).astype(bool)

fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(stockouts_data.index, stockouts_data['sup_prob'],
             color=custom_palette[1], linewidth=0.9, label='CatBoost P(anomaly)')
axes[0].axhline(best_thresh, color=custom_palette[2], linestyle='--', linewidth=1.5,
                label=f'Threshold ({best_thresh:.2f})')
axes[0].set_ylabel('Probability')
axes[0].legend()

axes[1].plot(stockouts_data.index, stockouts_data['sales'],
             color=custom_palette[0], linewidth=0.8, label='Sales')

tp_sup_df = stockouts_data[is_det_sup &  is_true]
fp_sup_df = stockouts_data[is_det_sup & ~is_true]

axes[1].scatter(tp_sup_df.index, tp_sup_df['sales'],
                color=custom_palette[2], s=60, zorder=5,
                marker='o', label=f'TP ({len(tp_sup_df)})')
if len(fp_sup_df) > 0:
    axes[1].scatter(fp_sup_df.index, fp_sup_df['sales'],
                    color=custom_palette[4], s=60, zorder=5,
                    marker='x', label=f'FP ({len(fp_sup_df)})')

axes[1].set_ylabel('Sales')
axes[1].set_xlabel('Date')
axes[1].legend()
fig.suptitle(
    f'CatBoost Supervised | Precision={prec_sup:.2f}, Recall={rec_sup:.2f}'
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: final comparison table
#| tbl-cap: "17.25: All methods comparison"
#| message: false
#| echo: false
#| eval: false

all_results = pd.DataFrame({
    'Method': [
        'Z-score Profile',
        'IF: Raw Features',
        'IF: Residuals',
        'INNE: Residuals',
        'LOF: Residuals',
        'CatBoost (supervised, tuned threshold)'
    ],
    'Type': [
        'Unsupervised', 'Unsupervised', 'Unsupervised',
        'Unsupervised', 'Unsupervised', 'Supervised'
    ],
    'Precision': [precision, prec_iu, prec_if, prec_eif, prec_lof, prec_sup],
    'Recall':    [recall,    rec_iu,  rec_if,  rec_eif,  rec_lof,  rec_sup],
})
all_results['F1'] = (
    2 * all_results['Precision'] * all_results['Recall'] 
    / (all_results['Precision'] + all_results['Recall'])
).round(2)
all_results['Precision'] = all_results['Precision'].round(2)
all_results['Recall'] = all_results['Recall'].round(2)

print(all_results.to_markdown(index=False))