
# if muiltiple GPU use $env:CUDA_VISIBLE_DEVICES = "0" in powershell or the equivolent in bash first to avoid multiprocessing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
from time import time
import matplotlib.pyplot as plt
from itertools import product,cycle

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
import torch
torch.manual_seed(243)

from functools import partial
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mase, mape, smape
import warnings
import seaborn as sns
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Simple line plot of time series - credit to Konrad Banachewics
def plot_series(df, xname):
    xd = df.loc[df.unique_id == xname ]
    airname = xd['Description'].iat[0]
    xd[['ds', 'y']].set_index('ds').plot(title = airname, xlabel = 'Time', color='orange')
    plt.tight_layout()
    plt.show()

# Plot histogram 
def plot_metric_distribution(eval_df, metric_names, model_names, figsize=(15, 10)):
    # Convert inputs to lists if they're strings
    metric_names = [metric_names] if isinstance(metric_names, str) else metric_names
    model_names = [model_names] if isinstance(model_names, str) else model_names
    
    plt.figure(figsize=figsize)
    
    for model in model_names:
        metric_values = eval_df[eval_df['metric'] == metric_names[0]][model]
        median_value = metric_values.median()
        plt.hist(metric_values, bins=50, alpha=0.5, label=f'{model} (median={median_value:.2f})', color='Orange')
        plt.axvline(median_value, color='black', linestyle='--', alpha=0.5)
    
    plt.xlabel(metric_names[0].upper())
    plt.ylabel('Count')
    plt.title(f'Distribution of {metric_names[0].upper()} across series') 
    plt.text(0.95, 0.95, metric_names[0].upper(), 
             transform=plt.gca().transAxes, 
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df, metric='smape', figsize=(12, 8)):  # Changed from (6, 8) to (12, 8)
    # Define a larger set of shapes and colors that we can cycle through
    shapes = ["o", "s", "*", "^", "D", "v", "<", ">", "p", "h"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", 
              "tab:purple", "tab:brown", "tab:pink", "tab:gray", 
              "tab:olive", "tab:cyan"]
    
    # Create cycle iterators
    style_cycle = cycle(product(shapes, colors))

    plt.figure(figsize=figsize, dpi=100)
    
    # Convert times to seconds (they're in minutes in the DataFrame)
    times_seconds = results_df['time'] * 60
    
    # Get metric values and determine if it needs percentage conversion
    metric_values = results_df[metric]
    if metric.lower() in ['mape', 'smape']:
        metric_values = metric_values * 100
        ylabel = f"median {metric.upper()} over all series (%)"
    else:
        ylabel = f"median {metric.upper()} over all series"
    
    # Calculate axis limits
    metric_range = metric_values.max() - metric_values.min()
    y_min = max(0, metric_values.min() - 0.1 * metric_range)
    y_max = metric_values.max() + 0.1 * metric_range
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        style = next(style_cycle)
        plt.semilogx(
            [times_seconds.iloc[i]],
            [metric_values.iloc[i]],
            style[0],  # shape
            color=style[1],  # color
            label=row['model'],
            markersize=13,
        )
    
    plt.xlabel("elapsed time [s]")
    plt.ylabel(ylabel)
    plt.ylim(y_min, y_max)
    plt.legend(bbox_to_anchor=(1.4, 1.0), frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Set forecast horizon
fh = 18

# Load processed air traffic data
air_data = (pd.read_csv('data\\international_report_departures.csv')
            .rename({'data_dte':'ds', 'airlineid':'unique_id', 'Total':'y'}, axis=1)
            .assign(ds=lambda x: pd.to_datetime(x['ds']) + pd.offsets.MonthEnd(0))
            .groupby(['ds', 'unique_id'])['y']
            .sum()
            .reset_index()
            .sort_values(['unique_id', 'ds'])
            .reset_index(drop=True))

air_data['ds'] = pd.to_datetime(air_data['ds'])

# Truncate series to 2019-12-31
air_data = air_data[air_data['ds'] <= '2019-12-31'].copy()

# map the identifiers to proper names
air_codes = pd.read_csv('data\\airline_codes.csv')
air_data = pd.merge(left = air_data, right = air_codes, left_on = 'unique_id', right_on = 'Code')
air_data.drop('Code', axis = 1, inplace = True)
air_data.head(5)

# list of names 
names_list = air_data.unique_id.unique()

plot_series(air_data, names_list[0])

# remove short series
min_size = 36

cleaned_series = []
for uid in air_data['unique_id'].unique():
    series_data = air_data[air_data['unique_id'] == uid].copy()
    
    # Resample to monthly frequency
    series_data = (series_data.set_index('ds')
                   .resample('1ME')['y']
                   .sum()
                   .clip(lower=1)
                   .reset_index())
    
    # Add back the unique_id
    series_data['unique_id'] = uid
    
    if len(series_data) >= (min_size + fh):
        cleaned_series.append(series_data)

air_data = pd.concat(cleaned_series)

# Check for gaps in dates for each series
for uid in air_data['unique_id'].unique():
    series_dates = air_data[air_data['unique_id'] == uid]['ds'].sort_values()
    # Convert the date differences to months
    date_diff = series_dates.diff().dt.total_seconds() / (30.44 * 24 * 60 * 60)  # approximate months
    gaps = date_diff[date_diff > 1.1]  # allowing some flexibility for different month lengths
    if not gaps.empty:
        print(f"Series {uid} has gaps at:")
        for idx, gap in gaps.items():
            print(f"  {series_dates[idx]} (gap of {gap:.1f} months)")

# Split train/test
air_train = []
air_test = []

for uid in air_data['unique_id'].unique():
    series_data = air_data[air_data['unique_id'] == uid].copy()
    series_data = series_data.sort_values('ds')  # Ensure data is sorted by date

    # Ensure the series is long enough before splitting
    if len(series_data) >= (min_size + fh):
        air_train.append(series_data.iloc[:-fh])  # Train: all but last fh points
        air_test.append(series_data.iloc[-fh:])   # Test: last fh points (newest dates)

# Combine train/test data
air_train = pd.concat(air_train, axis=0)
air_test = pd.concat(air_test, axis=0)

# Validate test set lengths
test_lengths = air_test.groupby('unique_id').size()
print("All series length 18:", all(test_lengths == fh))  # True is what we want
if not all(test_lengths == fh):
    print("Series with incorrect length:")
    print(test_lengths[test_lengths != fh])

# Set up error metrics
fcst_mase = partial(mase, seasonality=12)
metrics = [mae, mse, rmse, fcst_mase, mape, smape]
model_results = []
model_detailed_metrics = {}

# NBEATS
nbeats_config = NBEATS(
    h=18,
    input_size=fh *2,  # sum=36
    n_blocks=[1]*20,
    mlp_units=[[256, 256]] *3,
    stack_types=['identity', 'trend', 'seasonality'],
    learning_rate=1e-3,
    batch_size=1024,
    random_seed=243,
    max_steps=10,
   # early_stop_patience_steps=5,
    val_check_steps=100,
    scaler_type='standard'
)



# Initialize NeuralForecast with NBEATS
nbeats_model = NeuralForecast(
    models=[nbeats_config],
    freq='ME'
)

# Train NBEATS
start_time = time()
nbeats_model.fit(df=air_train) #, val_size = 6
nbeats_time = time() - start_time

# Generate NBEATS forecasts
nbeats_preds = nbeats_model.predict(futr_df=air_test)
nbeats_preds = nbeats_preds.reset_index()

# Prepare for evaluation
nbeats_eval_df = nbeats_preds.merge(
    air_test[['unique_id', 'ds', 'y']],
    on=['unique_id', 'ds'],
    how='left'
)

# Calculate NBEATS metrics
nbeats_metrics = evaluate(
    nbeats_eval_df,
    models=['NBEATS'],
    metrics=metrics,
    train_df=air_train,
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

plot_metric_distribution(nbeats_metrics, metric_names='smape', model_names='NBEATS')

# NHITS
nhits_config = NHITS(
    h=fh,
    input_size=30,
    stack_types=['identity', 'identity', 'identity'],
    n_blocks=[1, 1, 1],
    mlp_units=3 * [[512, 512]],
    n_pool_kernel_size=[2, 2, 1],
    n_freq_downsample=[4, 2, 1],
    learning_rate=1e-3,
    batch_size=64,
    windows_batch_size=1024,
    max_steps=1000,
    random_seed=243,
    scaler_type='standard',
)

# Initialize NeuralForecast with NHITS
nhits_model = NeuralForecast(
    models=[nhits_config],
    freq='ME'
)

# Train NHITS
start_time = time()
nhits_model.fit(df=air_train)
nhits_time = time() - start_time

# Generate NHITS forecasts
nhits_preds = nhits_model.predict(futr_df=air_test)
nhits_preds = nhits_preds.reset_index()

# Prepare for evaluation
nhits_eval_df = nhits_preds.merge(
    air_test[['unique_id', 'ds', 'y']],
    on=['unique_id', 'ds'],
    how='left'
)

# Calculate NHITS metrics
nhits_metrics = evaluate(
    nhits_eval_df,
    models=['NHITS'],
    metrics=metrics,
    train_df=air_train,
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

ml_results_df = pd.read_csv("example_notebooks//chp11_notebooks//data//chp11_gfm_median_ml_error_metrics.csv")

# Add NHITS results to ml_results_df
nhits_result = {
    'model': 'NHITS',
    'time': nhits_time / 60,
    **nhits_metrics.groupby('metric')['NHITS'].median().to_dict()
}
ml_results_df = pd.concat([ml_results_df, pd.DataFrame([nhits_result])], ignore_index=True)

# Add NBEATS results to ml_results_df
nbeats_result = {
    'model': 'NBEATS',
    'time': nbeats_time / 60,
    **nbeats_metrics.groupby('metric')['NBEATS'].median().to_dict()
}
ml_results_df = pd.concat([ml_results_df, pd.DataFrame([nbeats_result])], ignore_index=True)

ml_detailed_metrics = pd.read_csv('example_notebooks\\chp11_notebooks\\data\\chp11_gfm_ml_detailed_metrics.csv')
ml_detailed_metrics = (ml_detailed_metrics
    .merge(nhits_metrics, on=['unique_id', 'metric'], how='inner')
    .merge(nbeats_metrics, on=['unique_id', 'metric'], how='inner')
    .copy()
    )

# Display original DataFrame
ml_results_df

# Update each metric's median values from ml_detailed_metrics for all models due to dropping some series
metrics = ['mae', 'mase', 'mape', 'mse', 'rmse', 'smape']

for metric in metrics:
    metric_medians = ml_detailed_metrics[ml_detailed_metrics['metric'] == metric][
        ['linear', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor', 'NHITS', 'NBEATS']
    ].median()
    
    # Update values for each model
    for model in metric_medians.index:
        ml_results_df.loc[ml_results_df['model'] == model, metric] = metric_medians[model]

# Save updated versions
# ml_results_df.to_csv('example_notebooks/chp11_notebooks/data/chp11_gfm_median_ml_error_metrics.csv', index=False)
# ml_detailed_metrics.to_csv('example_notebooks/chp11_notebooks/data/chp11_gfm_ml_detailed_metrics.csv', index=False)



plot_model_comparison(ml_results_df, metric='rmse')


#####################################################################################################################
# PRETRAINED time series
m4_data = pd.read_csv("C:\\Users\\retailexp\\Documents\\TS Book Data\M4-Monthly-train.csv")
m4_data.head(3)
m4_meta = pd.read_csv("C:\\Users\\retailexp\\Documents\\TS Book Data\M4-info.csv")
m4_meta = m4_meta.loc[m4_meta.SP == "Monthly"].copy()

min_size = 48
horizon = 18  

def process_m4_data(m4_data, m4_meta, min_size=48, horizon=18):
    start_date = pd.Timestamp(m4_meta.loc[m4_meta["M4id"] == "M1", "StartingDate"].values[0])
    
    train_series = []
    test_series = []
    
    for idx, row in m4_data.iterrows():
        values = row.iloc[1:].dropna().values.astype('float32')
        
        if len(values) > min_size + horizon:
            dates = pd.date_range(start=start_date, periods=len(values), freq='ME')
            
            # Split into train/test
            train_dates = dates[:-horizon]
            test_dates = dates[-horizon:]
            train_values = values[:-horizon]
            test_values = values[-horizon:]
            
            # Create train DataFrame
            train_df = pd.DataFrame({
                'ds': train_dates,
                'y': train_values,
                'unique_id': row[0]
            })
            
            # Create test DataFrame
            test_df = pd.DataFrame({
                'ds': test_dates,
                'y': test_values,
                'unique_id': row[0]
            })
            
            train_series.append(train_df)
            test_series.append(test_df)
    
    return (pd.concat(train_series, axis=0).reset_index(drop=True),
            pd.concat(test_series, axis=0).reset_index(drop=True))

# Process M4 data
m4_train, m4_test = process_m4_data(m4_data, m4_meta)


# NBEATS
nbeats_config = NBEATS(
    h=18,
    input_size= fh *2,  # sum=36
    n_blocks=[1]*20,
    mlp_units=[[256, 256]] *3,
    stack_types=['identity', 'trend', 'seasonality'],
    learning_rate=1e-3,
    batch_size=1024,
    random_seed=243,
    max_steps=100,
    windows_batch_size=10,  
    val_check_steps=100,
    scaler_type='standard'
)



# Initialize NeuralForecast with NBEATS
nbeats_model = NeuralForecast(
    models=[nbeats_config],
    freq='ME'
)

# Train NBEATS
start_time = time()
nbeats_model.fit(df=m4_train) #, val_size = 6
nbeats_time = time() - start_time

# Generate NBEATS forecasts
nbeats_preds = nbeats_model.predict(df=air_train, futr_df=air_test) # provide train and test to get predictions on diff domain/set
nbeats_preds = nbeats_preds.reset_index()

# Prepare for evaluation
nbeats_eval_df = nbeats_preds.merge(
    air_test[['unique_id', 'ds', 'y']],
    on=['unique_id', 'ds'],
    how='left'
)

# Calculate NBEATS metrics
nbeats_metrics = evaluate(
    nbeats_eval_df,
    models=['NBEATS'],
    metrics=metrics,
    train_df=air_train,
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

plot_metric_distribution(nbeats_metrics, metric_names='smape', model_names='NBEATS')


# NHITS
nhits_config = NHITS(
    h=fh,
    input_size=30,
    stack_types=['identity', 'identity', 'identity'],
    n_blocks=[1, 1, 1],
    mlp_units=3 * [[512, 512]],
    n_pool_kernel_size=[2, 2, 1],
    n_freq_downsample=[4, 2, 1],
    learning_rate=1e-3,
    batch_size=64,
    windows_batch_size=1024,
    max_steps=1000,
    random_seed=243,
    scaler_type='standard',
)

# Initialize NeuralForecast with NHITS
nhits_model = NeuralForecast(
    models=[nhits_config],
    freq='ME'
)

# Train NHITS
start_time = time()
nhits_model.fit(df=m4_train)
nhits_time = time() - start_time

# Generate NHITS forecasts
nhits_preds = nhits_model.predict(df=air_train, futr_df=air_test)
nhits_preds = nhits_preds.reset_index()

# Prepare for evaluation
nhits_eval_df = nhits_preds.merge(
    air_test[['unique_id', 'ds', 'y']],
    on=['unique_id', 'ds'],
    how='left'
)

# Calculate NHITS metrics
nhits_metrics = evaluate(
    nhits_eval_df,
    models=['NHITS'],
    metrics=metrics,
    train_df=air_train,
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

# Add NHITS results to ml_results_df
nhits_result = {
    'model': 'NHITS_m4_pretrained',
    'time': nhits_time / 60,
    **nhits_metrics.groupby('metric')['NHITS'].median().to_dict()
}
ml_results_df = pd.concat([ml_results_df, pd.DataFrame([nhits_result])], ignore_index=True)

# Add NBEATS results to ml_results_df
nbeats_result = {
    'model': 'NBEATS_m4_pretrained',
    'time': nbeats_time / 60,
    **nbeats_metrics.groupby('metric')['NBEATS'].median().to_dict()
}
ml_results_df = pd.concat([ml_results_df, pd.DataFrame([nbeats_result])], ignore_index=True)

# Add suffix to columns in the metrics dataframes
nhits_metrics = nhits_metrics.rename(columns={'NHITS': 'NHITS_m4_pretrain'})
nbeats_metrics = nbeats_metrics.rename(columns={'NBEATS': 'NBEATS_m4_pretrain'})

ml_detailed_metrics = (ml_detailed_metrics
    .merge(nhits_metrics, on=['unique_id', 'metric'], how='inner')
    .merge(nbeats_metrics, on=['unique_id', 'metric'], how='inner')
    .copy()
    )

#ml_results_df.to_csv('example_notebooks/chp11_notebooks/data/chp11_gfm_median_ml_error_metrics.csv', index=False)
#ml_detailed_metrics.to_csv('example_notebooks/chp11_notebooks/data/chp11_gfm_ml_detailed_metrics.csv', index=False)

stats_results_df = pd.read_csv("example_notebooks\chp11_notebooks\data\chp11_gfm_median_error_metrics.csv")

# Concatenate stats_results_df to ml_results_df
ml_results_df = pd.concat([stats_results_df, ml_results_df], ignore_index=True)

pd.set_option('display.float_format', '{:.3f}'.format)
ml_results_df.sort_values(by=['rmse'])

plot_model_comparison(ml_results_df, metric='rmse')

pd.read_csv('C:\\Users\\retailexp\\Documents\\GitHub\\tsfwpt\\example_notebooks\\chp11_notebooks\\data\\chp11_gfm_detailed_metrics.csv')

