# If you have multiple GPU you may need to set it to one to avoid issues with training strategy
# apply %env CUDA_VISIBLE_DEVICES=0 in cmd via ipython
# $env:CUDA_VISIBLE_DEVICES = "0" # apply in powershell


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.set_device(0)


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, Autoformer, TFT, LSTM, PatchTST, NLinear #, TimesNet
from neuralforecast.losses.pytorch import MAE, MSE, SMAPE, RMSE, MASE
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mase, mape, smape
from functools import partial
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, Naive

# Define forecasting horizon and model parameters
h = 28  # 28 day forecast horizon
max_steps = 300  # Maximum training iterations
early_stop_patience= 25

# Structure data with required columns
df = pd.read_csv('C:\\Users\\retailexp\\Documents\\GitHub\\tsfwpt\\data\\M5_t20_ABC.csv')
df['timestamp'] = pd.to_datetime(df['date'])
df = df.rename(columns={
    'id': 'unique_id',  # Series identifier
    'date': 'ds',
    'sold': 'y',        # Target variable
    'sell_price': 'price'  # Static/dynamic feature
})

df['ds'] = pd.to_datetime(df['ds'])

df = df[['unique_id', 'ds', 'y', 'price']].copy()

df.head()

df.dtypes

# Define cut point (remove 28 days), and forecast horizon
cutpoint = 28 
h = 90

# Group by unique_id and apply the cutoff
df = df.groupby('unique_id').apply(lambda x: x.iloc[:-cutpoint]).reset_index(drop=True).copy()

print("DataFrame, check last 28 days removed")
print(df.tail())

# Create validation cutoff point
cutoff_date = df['ds'].max() - pd.Timedelta(days=90)

# Split maintaining temporal integrity
train_df = df[df['ds'] <= cutoff_date].reset_index(drop=True)
val_df = df[df['ds'] > cutoff_date].reset_index(drop=True)

# Select a few unique_ids to plot
unique_ids_to_plot = df['unique_id'].unique()[:3]

plt.figure(figsize=(15, 10))

for i, unique_id in enumerate(unique_ids_to_plot, 1):
    plt.subplot(len(unique_ids_to_plot), 1, i)
    plt.plot(train_df[train_df['unique_id'] == unique_id]['ds'], train_df[train_df['unique_id'] == unique_id]['y'], label='Train')
    plt.plot(val_df[val_df['unique_id'] == unique_id]['ds'], val_df[val_df['unique_id'] == unique_id]['y'], label='Validation')
    plt.axvline(cutoff_date, color='r', linestyle='--', label='Cutoff Date')
    plt.title(f'Time Series for {unique_id}')
    plt.xlabel('Date')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()


models = [
    NHITS(
        input_size=90,  # 3 months of history
        h=h,
        max_steps=max_steps,
        scaler_type='standard',  # Standardize across all series
        loss=MAE(),  # Using Mean Absolute Error
        hist_exog_list= None,      # Historic exogenous variable list
        futr_exog_list=['price'],  # Future exogenous list 
        stat_exog_list=None,       # Static exogenous list
        early_stop_patience_steps=early_stop_patience,
    ),
    Autoformer(
        input_size=90,
        h=h, 
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),
        hist_exog_list= None,      # Historic exogenous variable list
        futr_exog_list=['price'],  # Future exogenous list 
        stat_exog_list=None, 
        early_stop_patience_steps=early_stop_patience,
    ),
    NLinear(  # Simple but effective baseline
        input_size=90,
        h=h,
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),
        early_stop_patience_steps=early_stop_patience,
    ),
    PatchTST(  # Strong performer on M5
        input_size=90,
        h=h,
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),     
        early_stop_patience_steps=early_stop_patience,
    ),
    NBEATSx(  # Enhanced NBEATS with exogenous support
        input_size=90,
        h=h,
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),
        hist_exog_list= None,      # Historic exogenous variable list
        futr_exog_list=['price'],  # Future exogenous list 
        stat_exog_list=None, 
        early_stop_patience_steps=early_stop_patience,
    ),
    TFT(  # Good with exogenous variables
        input_size=90,
        h=h,
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),
        hist_exog_list= None,      # Historic exogenous variable list
        futr_exog_list=['price'],  # Future exogenous list 
        stat_exog_list=None, 
        early_stop_patience_steps=early_stop_patience,
    ),
    LSTM(  # Strong baseline for temporal dependencies
        input_size=90,
        h=h,
        max_steps=max_steps,
        scaler_type='standard',
        loss=MAE(),
        hist_exog_list= None,      # Historic exogenous variable list
        futr_exog_list=['price'],  # Future exogenous list 
        stat_exog_list=None, 
        early_stop_patience_steps=early_stop_patience,
    )
]

# Initialize global forecasting framework
nf = NeuralForecast(
    models=models,
    freq='D',  # Daily frequency
    #normalizer=None  # Let individual models handle scaling
)

nf.fit(
    df=train_df,
    val_size=90,  # Validation window size
    verbose=True
)

# Generate predictions with validation data
predictions = nf.predict(futr_df=val_df)

# Naive model comparison - always do this! 
base_models = [
    SeasonalNaive(season_length=7),  # Weekly seasonality
    Naive()
]

sf = StatsForecast(
    df=train_df,
    models=base_models,
    freq='D',
    n_jobs=-1
)

# Generate naive predictions
naive_predictions = sf.forecast(h=h)

eval_df = predictions.merge(
    naive_predictions, 
    on=['unique_id', 'ds'], 
    how='left'
    ).merge(
        val_df[['unique_id', 'ds', 'y']], 
        on=['unique_id', 'ds'], 
        how='left'
    )

fcst_mase = partial(mase, seasonality=7)
# Get metrics for individual series
series_metrics = evaluate(
    eval_df, 
    models=['NHITS', 'Autoformer','NLinear','PatchTST','NBEATSx','TFT','LSTM','SeasonalNaive', 'Naive'], # Specify your model names (in cols of eval_df)
    metrics=[mae, mse, rmse, fcst_mase, mape, smape],  
    train_df=train_df,      
    id_col='unique_id',
    time_col='ds',
    target_col='y'
)

# Get aggregated metrics across all series by model
agg_metrics = evaluate(
    eval_df, 
    models=['NHITS', 'Autoformer','NLinear','PatchTST','NBEATSx','TFT','LSTM', 'SeasonalNaive', 'Naive'], # Specify your model names (in cols of eval_df)
    metrics=[mae, mse, rmse, fcst_mase, mape, smape],  
    train_df=train_df,
    id_col='unique_id', 
    time_col='ds',
    target_col='y',
    agg_fn='mean'
)

print("\nOverall Metrics:")
print(agg_metrics)

print("\nPer-Series Metrics (first 5):")
print(series_metrics.head())

# Overall performance by ranking
agg_metrics_subset = agg_metrics.loc[agg_metrics['metric'].isin(['mae', 'mse', 'rmse', 'mase'])] # remove APE metrics due to how they favour under or over forecasting.

# Calculate ranks for each metric (lower is better)
ranks = agg_metrics_subset.set_index('metric').rank(axis=1)
avg_rank = ranks.mean()

rank_summary = pd.DataFrame({
    'Model': avg_rank.index,
    'Average Rank': avg_rank.values
}).sort_values('Average Rank')

print("\nOverall Model Rankings (Aggregate Metrics):")
print(rank_summary)

# For series-level metrics - count wins per metric
def count_best_performances(series_metrics):
    wins = {}
    metrics = ['mae', 'mse', 'rmse', 'mase']
    
    for metric in metrics:
        # Get model names that achieve best performance for each series
        metric_data = series_metrics[series_metrics['metric'] == metric]
        model_columns = [col for col in metric_data.columns if col not in ['metric', 'unique_id']]
        
        best_models = metric_data.groupby('unique_id')[model_columns].apply(
            lambda x: x.idxmin(axis=1)
        )
        
        # Count wins for each model
        win_counts = best_models.value_counts()
        wins[metric] = win_counts
    
    # Convert to DataFrame
    wins_df = pd.DataFrame(wins).fillna(0)
    wins_df['total_wins'] = wins_df.sum(axis=1)
    
    return wins_df

win_counts = count_best_performances(series_metrics)

# Ensure indices match before merging
win_counts = win_counts.reset_index()
win_counts = win_counts.rename(columns={'index': 'Model'})

# Create final summary
final_summary = pd.merge(
    rank_summary, 
    win_counts[['Model', 'total_wins']], 
    on='Model', 
    how='outer'
).fillna(0)

final_summary = final_summary.sort_values('Average Rank')
final_summary['Wins Rank'] = final_summary['total_wins'].rank(ascending=False)

print("\nFinal Model Performance Summary:")
print(final_summary)

val_df.unique_id.nunique()

############### Relative metrics #####################

def rel_rmse(y_true, y_pred, y_naive):
    """
    Calculate Relative RMSE using seasonal naive as baseline
    Values < 1 indicate better performance than seasonal naive
    Values > 1 indicate worse performance than seasonal naive
    RelRMSE = √(MSE(pred) / MSE(naive))
    """
    mse_pred = np.mean((y_true - y_pred) ** 2)
    mse_naive = np.mean((y_true - y_naive) ** 2)
    return np.sqrt(mse_pred / mse_naive)

def rel_mae(y_true, y_pred, y_naive):
    """
    Calculate Relative MAE using seasonal naive as baseline
    Values < 1 indicate better performance than seasonal naive
    Values > 1 indicate worse performance than seasonal naive
    RelMAE = MAE(pred) / MAE(naive)
    """
    mae_pred = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    return mae_pred / mae_naive

rel_rmse_val = rel_rmse(
                eval_df['y'].values,
                eval_df['NBEATSx'].values,
                eval_df['SeasonalNaive'].values
            )

rel_mae_val = rel_mae(
                eval_df['y'].values,
                eval_df['NBEATSx'].values,
                eval_df['SeasonalNaive'].values
            )

# You can train a global model in NeuralForecast by following these steps:
# 
# 1. First, define your hyperparameter configuration. Here's an example for NHITS model  [(1)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/getting_started_complete.html#4-train-multiple-models-for-many-series) :
# 
# ```python
# def config_nhits(trial):
#     return {
#         "input_size": trial.suggest_categorical(          # Length of input window
#             "input_size", (48, 48*2, 48*3)                
#         ),                                                
#         "start_padding_enabled": True,                                          
#         "n_blocks": 5 * [1],                              # Length of input window
#         "mlp_units": 5 * [[64, 64]],                      # Length of input window
#         "n_pool_kernel_size": trial.suggest_categorical(  # MaxPooling Kernel size
#             "n_pool_kernel_size",
#             (5*[1], 5*[2], 5*[4], [8, 4, 2, 1, 1])
#         ),     
#         "learning_rate": trial.suggest_float(             # Initial Learning rate
#             "learning_rate",
#             low=1e-4,
#             high=1e-2,
#             log=True,
#         )
#     }
# ```
# 2. Initialize the NeuralForecast model with your configuration :
# 
# ```python
# nf = NeuralForecast(
#     models=[
#         AutoNHITS(h=48, config=config_nhits, loss=MQLoss(), backend='optuna', num_samples=5)
#     ],
#     freq=1,
# )
# ```
# 3. Train the model using the fit method :
# 
# ```python
# nf.fit(df=Y_df)
# ```
# Note that all NeuralForecast models are "global" because they train with all series from the input DataFrame data, though the optimization objective is momentarily "univariate" as it doesn't consider interaction between output predictions across time series  [(2)](https://nixtlaverse.nixtla.io/neuralforecast/models.html) .

 
# For NHITS, here are some recommended additional parameters based on the documentation:
# 
# ```python
# NHITS(
#     input_size=90,
#     h=h,
#     max_steps=max_steps,
#     scaler_type='standard',
#     loss=MAE(),
#     n_pool_kernel_size=[2, 2, 2],  # MaxPool's Kernel size
#     n_freq_downsample=[24, 12, 1], # Interpolation expressivity ratios  
#     n_blocks=[1, 1, 1],            # Blocks per stack
#     mlp_units=[[512, 512], [512, 512], [512, 512]], # Hidden layers
#     interpolation_mode='linear',    # Type of interpolation
#     val_check_steps=100            # Validation frequency
# )
# ```
#  [(1)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_nhits.html#4-evaluate-results) 
# 
# For Autoformer, the documentation suggests these additional parameters:
# 
# ```python
# Autoformer(
#     input_size=90,
#     h=h,
#     max_steps=max_steps, 
#     scaler_type='standard',
#     loss=MAE(),
#     hidden_size=128,        # Size of embeddings/encoders
#     n_head=4,              # Number of attention heads
#     dropout=0.05,          # Dropout rate
#     activation='gelu',     # Activation function
#     encoder_layers=2,      # Number of encoder layers
#     decoder_layers=1       # Number of decoder layers
# )
# ```
#  [(2)](https://nixtlaverse.nixtla.io/neuralforecast/models.autoformer.html#2-autoformer) 
# 
# Regarding your input size of 90 for predicting 90 days ahead: Based on the NHITS paper implementation, they typically use an input size that is 5 times the forecast horizon for optimal results [(1)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_nhits.html#4-evaluate-results) . Given you're forecasting 90 days ahead, you might want to consider increasing your input_size to around 450 (5 * 90) to capture longer-term patterns, especially since you have 970 days of historical data available.
# 
# The documentation shows examples where they use `input_size = multiplier * horizon` where multiplier is typically 5. This allows the model to learn patterns across multiple seasonal cycles while maintaining computational efficiency.

#  
# 
# For M5-like datasets, NeuralForecast offers several powerful models specifically designed for global modeling. The key models include:
# 
# **Deep Learning Models:**
# - AutoNHITS (N-HITS architecture)
# - AutoTFT (Temporal Fusion Transformers)
#  [(1)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/comparing_methods.html) ,  [(2)](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html#neuralforecast) 
# 
# These models are "global" because they train on all series simultaneously while maintaining a univariate optimization objective  [(3)](https://github.com/Nixtla/neuralforecast/discussions/705) . This means one set of weights is used for all time series in your dataset .
# 
# **Key Features of NeuralForecast for Global Modeling:**
# - Broad collection of global models including MLP, LSTM, RNN, TCN, DilatedRNN, NBEATS, NHITS, ESRNN, TFT, Informer, PatchTST and HINT  [(2)](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html#neuralforecast) ,  [(4)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/comparing_methods.html#neuralforecast) 
# - Simple interface for training, forecasting and backtesting  [(2)](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/statisticalneuralmethods.html#neuralforecast) 
# - GPU acceleration support for improved computational speed For optimal results with global modeling, it's recommended to:
# - Use GPU acceleration when possible  [(5)](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/getting_started_complete.html) 
# - Leverage cross-validation to evaluate model performance - Consider adding static exogenous variables to capture group-specific dynamics  [(3)](https://github.com/Nixtla/neuralforecast/discussions/705) 

