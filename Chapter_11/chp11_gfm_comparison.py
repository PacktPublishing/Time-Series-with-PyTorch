import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,  
    AutoARIMA,      
    AutoETS,          
    Theta          
)
from functools import partial
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mase, mape, smape
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load processed air traffic data
air_data = pd.read_csv('data\\international_report_departures.csv').rename({'data_dte':'ds'}, axis=1)
air_data['ds'] = pd.to_datetime(air_data['ds'])
air_data = air_data.groupby(['ds', 'airlineid'])['Total'].sum().reset_index()
air_data = air_data.sort_values(['airlineid','ds']).reset_index(drop=True)
air_data = air_data.rename({'airlineid':'unique_id', 'Total':'y'}, axis=1) # our dependent varible 'Total' passengers we make y and our id for each series unique_id for nixtla libraries.
air_data['ds'] = air_data['ds'] - pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(0) # aggregations set date to end of month, but forecasts in nixtla set beginning of month

min_size = 48  
series_lengths = air_data.groupby('unique_id').size()
valid_series = series_lengths[series_lengths >= min_size].index
air_data = air_data[air_data['unique_id'].isin(valid_series)]

zero_counts = air_data[air_data['y'] == 0].groupby('unique_id').size()
print(zero_counts)

# Set forecast horizon
fh = 18

# Create train/test splits for each unique_id (lengths and times are different)
air_train = []
air_test = []

for uid in air_data['unique_id'].unique():
    series_data = air_data[air_data['unique_id'] == uid].copy()
    
    # Get cutoff date for this specific series
    series_cutoff = series_data['ds'].max() - pd.DateOffset(months=fh)
    
    # Split the series
    series_train = series_data[series_data['ds'] <= series_cutoff]
    series_test = series_data[series_data['ds'] > series_cutoff]
    
    air_train.append(series_train)
    air_test.append(series_test)

# Combine all train and test data
air_train = pd.concat(air_train, axis=0)
air_test = pd.concat(air_test, axis=0)

# 'Simple models' for baseline comparison
models = [
    SeasonalNaive(season_length=12),  # Monthly seasonality
    AutoARIMA(),
    AutoETS(),
    Theta()
]

fcst_mase = partial(mase, seasonality=12)
metrics = [mae, mse, rmse, fcst_mase, mape, smape]
model_results = []
model_detailed_metrics = {}


# Evaluate each model separately with fit time
for model in models:
    # Initialize single model StatsForecast
    sf_single = StatsForecast(
        models=[model],
        freq='ME',
        n_jobs=-1
    )
    
    # Time the fit
    start_time = time()
    sf_single.fit(df=air_train)
    fit_time = time() - start_time
    
    # Get forecasts
    fcst = sf_single.forecast(h=fh)
    
    # Get model name and rename forecast column to match what evaluate expects
    model_name = model.__class__.__name__
    fcst = fcst.rename(columns={model_name: model_name})  # This ensures column name matches
    
    fcst['ds'] = pd.to_datetime(fcst['ds']) + pd.offsets.MonthBegin(0)
    eval_df = fcst.merge(
                air_test[['unique_id', 'ds', 'y']],
                on=['unique_id', 'ds'],
                how='left')
    
    # Get detailed metrics
    series_metrics = evaluate(
        eval_df,
        models=[model_name],  # Note: Changed to list to match working example
        metrics=metrics,
        train_df=air_train,
        id_col='unique_id',
        time_col='ds',
        target_col='y'
    )

    model_detailed_metrics[model_name] = series_metrics
    
    # Calculate median metrics
    median_metrics = series_metrics.groupby('metric')[model_name].median()
    
    # Store results with fit time
    model_results.append({
        'model': model_name,
        'time': fit_time / 60,  # Convert to minutes
        **median_metrics.to_dict()  # Just unpack the median_metrics directly
    })

# Convert to DataFrame
results_df = pd.DataFrame(model_results)

plt.figure(figsize=(15, 10))
metrics_to_plot = ['smape']  # change to desired metric

for i, metric_name in enumerate(metrics_to_plot, 1):
    plt.subplot(len(metrics_to_plot), 1, i)
    
    for model_name, metrics_df in model_detailed_metrics.items():
        metric_values = metrics_df[metrics_df['metric'] == metric_name][model_name]
        plt.hist(metric_values, bins=50, alpha=0.5, label=model_name)
    
    plt.xlabel(metric_name.upper())
    plt.ylabel('Count')
    plt.title(f'Distribution of {metric_name.upper()} across series')
    plt.legend()

plt.tight_layout()
plt.show()
# Save results DataFrame as median_error_metrics
results_df.to_csv('chp11_gfm_median_error_metrics.csv', index=False)

# Convert model_detailed_metrics into a DataFrame
detailed_metrics_list = []

for model_name, metrics_df in model_detailed_metrics.items():
    metrics_df = metrics_df.rename(columns={model_name: 'value'})
    metrics_df['model'] = model_name
    detailed_metrics_list.append(metrics_df)

detailed_metrics_df = pd.concat(detailed_metrics_list, axis=0)

# Save detailed metrics DataFrame
detailed_metrics_df.to_csv('chp11_gfm_detailed_metrics.csv', index=False)

# GFM on airtraffic 
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, ExponentiallyWeightedMean, RollingMean
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utilsforecast.feature_engineering import time_features


# Train Linear Regression
linear_model = MLForecast(
    models={'linear': LinearRegression()},
    freq='ME', 
    lags=range(1,12),
    date_features=['month', 'year']
)
start_time = time()
linear_model.fit(
    df=air_train,
    id_col='unique_id', 
    time_col='ds',
    target_col='y',
    static_features=[]
)
linear_time = time() - start_time
linear_preds = linear_model.predict(h=fh)

# Train LightGBM
lgb_model = MLForecast(
    models=[lgb.LGBMRegressor(verbosity=-1,random_state=352, objective='mae')],
    freq='ME',
    lags=range(1,12),
    lag_transforms={
        1: [RollingMean(12), ExponentiallyWeightedMean(alpha=0.3)],
        12: [ExpandingMean()]
    },
    date_features=['month', 'year']
)
start_time = time()
lgb_model.fit(air_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
lgb_time = time() - start_time
lgb_preds = lgb_model.predict(h=fh)

xgb_model = MLForecast(
    models=[xgb.XGBRegressor(objective='reg:squarederror', random_state=352)],
    freq='ME',
    lags=range(1, 12),
    lag_transforms={
        1: [RollingMean(12), ExponentiallyWeightedMean(alpha=0.3)],
        12: [ExpandingMean()]
    },
    date_features=['month', 'year']
)
start_time = time()
xgb_model.fit(air_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
xgb_time = time() - start_time
xgb_preds = xgb_model.predict(h=fh)

rf_model = MLForecast(
    models=[RandomForestRegressor(random_state=352, n_jobs=-1)],
    freq='ME',
    lags=range(1, 12),
    lag_transforms={
        1: [RollingMean(7), ExponentiallyWeightedMean(alpha=0.3)],
        7: [ExpandingMean()]
    },
    date_features=['month', 'year']
)

start_time = time()
rf_model.fit(air_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
rf_time = time() - start_time
rf_preds = rf_model.predict(h=fh)


# Combine predictions together on unique_id and ds
all_preds = linear_preds.merge(lgb_preds, on=['unique_id', 'ds'])
all_preds = all_preds.merge(xgb_preds, on=['unique_id', 'ds'])
all_preds = all_preds.merge(rf_preds, on=['unique_id', 'ds'])


all_preds['ds'] = pd.to_datetime(all_preds['ds']) + pd.offsets.MonthBegin(0)
eval_df = all_preds.merge(
                air_test[['unique_id', 'ds', 'y']],
                on=['unique_id', 'ds'],
                how='left')
    
  
ml_model_names = [col for col in eval_df.columns if col not in ['ds', 'y', 'unique_id']]
# Get detailed metrics
ml_series_metrics = evaluate(
        eval_df,
        models=ml_model_names, 
        metrics=metrics,
        train_df=air_train,
        id_col='unique_id',
        time_col='ds',
        target_col='y'
    )

#model_detailed_metrics['linear', 'LGBMRegressor', 'RandomForestRegressor'] = ml_series_metrics
    
# Calculate median metrics
ml_median_metrics = ml_series_metrics.groupby('metric')[ml_model_names].median()
    
model_time_mapping = {
    'LGBMRegressor': 'lgb_time',
    'XGBRegressor': 'xgb_time',
    'RandomForestRegressor': 'rf_time'
}

# Store results with fit time
ml_model_results = []
for model_name in ml_model_names:
    time_var = model_time_mapping.get(model_name, f"{model_name.lower()}_time")
    if time_var in globals():
        ml_model_results.append({
            'model': model_name,
            'time': eval(time_var) / 60,  # Convert time to minutes
            **ml_median_metrics[model_name].to_dict()  # Extract median metrics for this model
        })
    else:
        print(f"Warning: Timing variable for {model_name} ({time_var}) is not defined.")

# Convert to DataFrame
ml_results_df = pd.DataFrame(ml_model_results)

# Save results DataFrame as median_error_metrics
ml_results_df.to_csv('chp11_gfm_median_ml_error_metrics.csv', index=False)

# Save individual series metrics DataFrame
ml_series_metrics.to_csv('chp11_gfm_ml_detailed_metrics.csv', index=False)

def plot_model_metric(metrics_df, model_col, metric_name='smape', figsize=(15, 10)):
    """
    Plot histogram for a single model's metric.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        The metrics DataFrame
    model_col : str
        Name of model column to plot
    metric_name : str
        Name of metric to plot (default: 'smape')
    """
    metric_data = metrics_df[metrics_df['metric'] == metric_name] *100
    median_value = metric_data[model_col].median() 
    
    plt.figure(figsize=figsize)
    plt.hist(metric_data[model_col].clip(upper=120), bins=50, color="orange")
    plt.xlabel(metric_name.upper())
    plt.ylabel('Count')
    plt.title(f'{model_col} - Median {metric_name.upper()}: {median_value:.3f}')
    plt.xlim(0, 120)
    plt.show()
    plt.tight_layout()

# Example usage:
plot_model_metric(ml_series_metrics, 'linear')
plot_model_metric(ml_series_metrics, 'LGBMRegressor')
plot_model_metric(ml_series_metrics, 'XGBRegressor')
plot_model_metric(ml_series_metrics, 'RandomForestRegressor')


from itertools import product,cycle

stats_median_df = pd.read_csv("example_notebooks\\chp11_notebooks\\data\\chp11_gfm_median_error_metrics.csv")

results_df = pd.concat([stats_median_df, ml_results_df], axis=0)



def plot_ml_comparison(results_df, figsize=(6, 6)):
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
    # Convert SMAPE to percentage (0-100 scale)
    smapes_pct = results_df['smape'] * 100
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        style = next(style_cycle)
        plt.semilogx(
            [times_seconds.iloc[i]],
            [smapes_pct.iloc[i]],
            style[0],  # shape
            color=style[1],  # color
            label=row['model'],
            markersize=13,
        )
    
    plt.xlabel("elapsed time [s]")
    plt.ylabel("median sMAPE over all series")
    plt.legend(bbox_to_anchor=(1.4, 1.0), frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Use the function
plot_ml_comparison(results_df)

