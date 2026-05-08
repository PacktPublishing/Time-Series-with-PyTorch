#!/usr/bin/env python3

import sys
import os

module_dir = r"C:\Users\retailexp\Documents\GitHub\tsfwpt\example_notebooks\chp11_notebooks\simple_gfm_pytorch"
if module_dir not in sys.path:
    sys.path.append(module_dir)

from dataset import TimeSeriesDataset
from train import GlobalForecastingModel, load_m5_data,SimpleProgress
from main import create_optimized_dataloaders
from get_predictions import (
    load_trained_model, 
    evaluate_model, 
    generate_predictions, 
    plot_predictions
)

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)

import matplotlib.pyplot as plt

# 1. Load data
path_to_data = "C:/Users/retailexp/Documents/GitHub/tsfwpt/data/M5_t20_ABC.csv"
train_dict, test_dict = load_m5_data(path_to_data)

# 2. Create Datasets and Dataloaders
train_loader, val_loader, test_loader = create_optimized_dataloaders(
    train_dict=train_dict,
    test_dict=test_dict,
    lookback=14,
    horizon=1,
    batch_size=32,
    num_workers=4,
    val_split=0.2
)


# 4. Define Three Different Integration Architectures
#    (Concat, Add, and Gated)


class ConcatIntegrationArchitecture(nn.Module):
    """Temporal output and embeddings are concatenated along feature dim."""
    def __init__(self, num_series, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.series_embeddings = nn.Embedding(num_series, hidden_dim)
        nn.init.xavier_uniform_(self.series_embeddings.weight)
        
        self.temporal_encoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, series_ids):
        features = self.feature_proj(x)
        temporal_out, _ = self.temporal_encoder(features)
        # final hidden state from LSTM
        lstm_output = temporal_out[:, -1, :]
        
        # get embedding
        series_embed = self.series_embeddings(series_ids.squeeze())
        # concat
        combined = torch.cat([lstm_output, series_embed], dim=1)
        return self.decoder(combined)


class AddIntegrationArchitecture(nn.Module):
    """Temporal output and embeddings are added element-wise."""
    def __init__(self, num_series, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.series_embeddings = nn.Embedding(num_series, hidden_dim)
        nn.init.xavier_uniform_(self.series_embeddings.weight)
        
        self.temporal_encoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, series_ids):
        features = self.feature_proj(x)
        temporal_out, _ = self.temporal_encoder(features)
        lstm_output = temporal_out[:, -1, :]
        
        series_embed = self.series_embeddings(series_ids.squeeze())
        # add
        combined = lstm_output + series_embed
        return self.decoder(combined)


class GatedIntegrationArchitecture(nn.Module):
    """Temporal output is gated before being combined with series embedding."""
    def __init__(self, num_series, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.series_embeddings = nn.Embedding(num_series, hidden_dim)
        nn.init.xavier_uniform_(self.series_embeddings.weight)
        
        self.temporal_encoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # gate layer that outputs gating factors
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, series_ids):
        features = self.feature_proj(x)
        temporal_out, _ = self.temporal_encoder(features)
        lstm_output = temporal_out[:, -1, :]
        
        series_embed = self.series_embeddings(series_ids.squeeze())
        # gating
        gating_factor = self.gate_fc(lstm_output)  # shape: [batch_size, hidden_dim]
        gated_output = gating_factor * lstm_output  # element-wise scaling
        # concat the gated LSTM output with the embedding
        combined = torch.cat([gated_output, series_embed], dim=1)
        return self.decoder(combined)


# ------------------------------------------------------------------------------
# 5. Define a Generic LightningModule that Accepts Any of the Above Architectures
# ------------------------------------------------------------------------------
class GlobalForecastingModel(L.LightningModule):
    def __init__(self, architecture, learning_rate=1e-3):
        super().__init__()
        self.model = architecture
        self.learning_rate = learning_rate
    
    def forward(self, x, series_ids):
        return self.model(x, series_ids)
    
    def _unscale_predictions(self, y_scaled, scaling_factors):
        """Safely unscale predictions to original scale."""
        center = scaling_factors[:, 0].unsqueeze(-1)
        scale = scaling_factors[:, 1].unsqueeze(-1)
        return y_scaled * scale + center
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        sid = batch['series_idx']
        y_hat = self(x, sid).squeeze(-1)
        loss = F.mse_loss(y_hat, y[:, 0])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        sid = batch['series_idx']
        scaling_factors = batch['scaling_factors']
        
        # Get predictions in scaled space
        y_hat = self(x, sid).squeeze(-1)
        
        # Calculate loss in scaled space
        val_loss = F.mse_loss(y_hat, y[:, 0])
        
        # Safely transform predictions and targets back to original scale
        y_hat_orig = self._unscale_predictions(y_hat, scaling_factors)
        y_orig = self._unscale_predictions(y[:, 0], scaling_factors)
        
        # Calculate metrics in original scale
        val_loss_orig = F.mse_loss(y_hat_orig, y_orig)
        
        # Log both losses
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        self.log('val_loss_original_scale', val_loss_orig, prog_bar=True, on_epoch=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        sid = batch["series_idx"]
        scaling_factors = batch['scaling_factors']
        
        # Get predictions in scaled space
        y_hat = self(x, sid).squeeze(-1)
        
        # Transform predictions and targets back to original scale
        y_hat_orig = self._unscale_predictions(y_hat, scaling_factors)
        y_orig = self._unscale_predictions(y[:, 0], scaling_factors)
        
        # Calculate loss in original scale
        test_loss = F.mse_loss(y_hat_orig, y_orig)
        
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


MAX_EPOCHS = 200  # or a larger number for more thorough training


# Example EarlyStopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',     # metric to monitor
    min_delta=1.00,         # minimum difference to consider an improvement
    patience=5,             # stop after 5 checks with no improvement
    verbose=True,
    mode='min'              # 'min' for loss, 'max' for accuracy
)

concat_checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='concat-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1, # keep best
    mode='min',
    save_on_train_epoch_end=True  # only save model at end of epoch
)

add_checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='add-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_on_train_epoch_end=True
)

gated_checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='gated-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_on_train_epoch_end=True
)

# (A) Concat Integration
concat_arch = ConcatIntegrationArchitecture(
    num_series=len(train_dict),
    input_dim=2,
    hidden_dim=64,
    num_layers=2,
    dropout=0.1
)
concat_model = GlobalForecastingModel(architecture=concat_arch, learning_rate=1e-3)

trainer_concat = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu',  # or 'cpu' if no GPU
    devices=1,
    precision='16-mixed',
    callbacks=[SimpleProgress(), early_stop_callback, concat_checkpoint_callback],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    min_epochs=20, 
)
print("\n[Training ConcatIntegrationArchitecture]")
trainer_concat.fit(concat_model, train_loader, val_loader)

# (B) Add Integration

add_arch = AddIntegrationArchitecture(
    num_series=len(train_dict),
    input_dim=2,
    hidden_dim=64,
    num_layers=2,
    dropout=0.1
)
add_model = GlobalForecastingModel(architecture=add_arch, learning_rate=1e-3)

trainer_add = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',
    callbacks=[SimpleProgress(), early_stop_callback, add_checkpoint_callback],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    min_epochs=20, 
)
print("\n[Training AddIntegrationArchitecture]")
trainer_add.fit(add_model, train_loader, val_loader)

# (C) Gated Integration
gated_arch = GatedIntegrationArchitecture(
    num_series=len(train_dict),
    input_dim=2,
    hidden_dim=64,
    num_layers=2,
    dropout=0.1
)
gated_model = GlobalForecastingModel(architecture=gated_arch, learning_rate=1e-3)

trainer_gated = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',
    callbacks=[SimpleProgress(), early_stop_callback, gated_checkpoint_callback],
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    min_epochs=20, 
)
print("\n[Training GatedIntegrationArchitecture]")
trainer_gated.fit(gated_model, train_loader, val_loader)

# direct save due to checkpointing error
trainer_concat.save_checkpoint("checkpoints/concat_final.ckpt")
trainer_add.save_checkpoint("checkpoints/add_final.ckpt")
trainer_gated.save_checkpoint("checkpoints/gated_final.ckpt")

# reload
concat_model_loaded = GlobalForecastingModel.load_from_checkpoint(
    "checkpoints/concat_final.ckpt", 
    architecture=ConcatIntegrationArchitecture(
        num_series=len(train_dict),
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ), 
    learning_rate=1e-3
)

add_model_loaded = GlobalForecastingModel.load_from_checkpoint(
    "checkpoints/add_final.ckpt", 
    architecture=AddIntegrationArchitecture(
        num_series=len(train_dict),
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ), 
    learning_rate=1e-3
)

gated_model_loaded = GlobalForecastingModel.load_from_checkpoint(
    "checkpoints/gated_final.ckpt", 
    architecture=GatedIntegrationArchitecture(
        num_series=len(train_dict),
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ), 
    learning_rate=1e-3
)

#test_trainer = L.Trainer(accelerator='gpu', devices=1)

# 7.4 Evaluate Each Model on the Test Set

#     For demonstration, we'll show just the test loss (no fancy metric loop).
test_result_concat = trainer_concat.test(concat_model_loaded, dataloaders=test_loader)
test_result_add = trainer_add.test(add_model_loaded, dataloaders=test_loader)
test_result_gated = trainer_gated.test(gated_model_loaded, dataloaders=test_loader)

print("\n======== Final Test Results ========")
print("ConcatIntegrationArchitecture ->", test_result_concat)
print("AddIntegrationArchitecture ->", test_result_add)
print("GatedIntegrationArchitecture ->", test_result_gated)

def generate_predictions(models, dataloader, device='cuda', include_naive=True):
    """
    Generate predictions for multiple models on the same dataset.
    
    Args:
        models: Dict of name -> model pairs
        dataloader: DataLoader containing test data
        device: Device to run predictions on
        
    Returns:
        Dict containing predictions and actuals for each series
    """
    results = {}
    
    # Initialize storage for each series
    for batch in dataloader:
        sid = batch['series_idx'].numpy()
        for s in sid:
            if s[0] not in results:
                results[s[0]] = {
                    'actuals': [],
                    'predictions': {model_name: [] for model_name in models.keys()}
                }
    
    # Generate predictions
    for model_name, model in models.items():
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['input'].to(device)
                y = batch['target']
                sid = batch['series_idx'].numpy()
                
                # Generate predictions
                preds = model(x, batch['series_idx'].to(device)).cpu()
                
                # Store predictions and actuals
                for i, series_id in enumerate(sid):
                    results[series_id[0]]['predictions'][model_name].append(preds[i].item())
                    if model_name == list(models.keys())[0]:  # Only store actuals once
                        results[series_id[0]]['actuals'].append(y[i, 0].item())
    
    return results

def add_naive_forecast(results, seasonality=7):
    """
    Add seasonal naive forecast that repeats the first week of data.
    
    Args:
        results: Dictionary containing predictions and actuals
        seasonality: Length of the seasonal pattern to repeat (e.g., 7 for weekly)
    """
    for series_data in results.values():
        # Get the first seasonal pattern
        seasonal_pattern = series_data['actuals'][:seasonality]
        
        # Create predictions by repeating the pattern
        naive_preds = []
        for i in range(len(series_data['actuals'])):
            # Use modulo to repeat the pattern
            pattern_index = i % seasonality
            naive_preds.append(seasonal_pattern[pattern_index])
            
        series_data['predictions']['naive'] = naive_preds
    return results

def calculate_metrics(results):
    """
    Calculate error metrics across all series.
    
    Args:
        results: Dict containing predictions and actuals
        
    Returns:
        Dict of metrics for each model
    """
    metrics = {}
    
    # Collect all predictions and actuals
    all_actuals = []
    all_predictions = {model_name: [] for model_name in list(results.values())[0]['predictions'].keys()}
    
    for series_data in results.values():
        all_actuals.extend(series_data['actuals'])
        for model_name in all_predictions.keys():
            all_predictions[model_name].extend(series_data['predictions'][model_name])
    
    # Convert to numpy arrays
    all_actuals = np.array(all_actuals)
    for model_name in all_predictions:
        all_predictions[model_name] = np.array(all_predictions[model_name])
    
    # Calculate metrics for each model
    for model_name, preds in all_predictions.items():
        mse = np.mean((preds - all_actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - all_actuals))
        mape = np.mean(np.abs((preds - all_actuals) / np.where(all_actuals != 0, all_actuals, 1))) * 100
        
        metrics[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    return metrics

def create_results_tables(results, metrics):
    """
    Create pandas DataFrames for detailed results analysis.
    
    Returns:
        Tuple of (summary_df, detailed_df)
    """
    # Create summary metrics table
    summary_data = []
    for model_name, model_metrics in metrics.items():
        row = {'Model': model_name}
        row.update(model_metrics)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Model')
    
    # Create detailed results table
    detailed_data = []
    for series_id, series_data in results.items():
        actuals = series_data['actuals']
        for i in range(len(actuals)):
            row = {
                'series_id': series_id,
                'timestep': i,
                'actual': actuals[i]
            }
            for model_name, preds in series_data['predictions'].items():
                row[f'{model_name}_pred'] = preds[i]
            detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    
    return summary_df, detailed_df

def plot_series_predictions(detailed_df, series_id, figsize=(12, 6)):
    """
    Plot actual vs predictions for a single series.
    
    Args:
        detailed_df: DataFrame containing all predictions and actuals
        series_id: ID of the series to plot
        figsize: Figure size (width, height)
    """
    # Filter for selected series
    series_data = detailed_df[detailed_df['series_id'] == series_id]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(series_data['timestep'], series_data['actual'], 
            'k-', label='Actual', linewidth=2)
    
    # Plot each model's predictions
    colors = {
        'naive': '#ba0f0f',
        'concat': '#003DFD',
        'add': '#b512b8',
        'gated': '#11a9ba'
    }
    
    # Plot predictions for each model
    for model in ['naive', 'concat', 'add', 'gated']:
        column = f'{model}_pred'
        if column in series_data.columns:
            ax.plot(series_data['timestep'], series_data[column],
                   color=colors[model],  
                   label=model.capitalize())
    
    # Customize plot
    ax.set_title(f'Series {series_id} - Actual vs Predicted', pad=20)
    ax.legend()
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def plot_multiple_series(detailed_df, series_ids, figsize=(15, 12)):
    """
    Plot predictions for multiple series.
    
    Args:
        detailed_df: DataFrame containing all predictions and actuals
        series_ids: List of series IDs to plot
        figsize: Figure size (width, height)
    """
    # Create subplots
    fig, axes = plt.subplots(len(series_ids), 1, figsize=figsize)
    if len(series_ids) == 1:
        axes = [axes]
    
    # Colors for different models
    colors = {
        'naive': '#ba0f0f',
        'concat': '#003DFD',
        'add': '#b512b8',
        'gated': '#11a9ba'
    }
    
    # Plot each series
    for ax, series_id in zip(axes, series_ids):
        series_data = detailed_df[detailed_df['series_id'] == series_id]
        
        # Plot actual values
        ax.plot(series_data['timestep'], series_data['actual'], 
                'k-', label='Actual', linewidth=2)
        
        # Plot each model's predictions
        for model in ['naive', 'concat', 'add', 'gated']:
            column = f'{model}_pred'
            if column in series_data.columns:
                ax.plot(series_data['timestep'], series_data[column],
                       color=colors[model], 
                       label=model.capitalize())
        
        ax.set_title(f'Series {series_id}', pad=10)
        ax.legend()
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    return fig

# Create dictionary of models
models = {
    'concat': concat_model_loaded,
    'add': add_model_loaded,
    'gated': gated_model_loaded
}

# Generate predictions
results = generate_predictions(models, test_loader)
results = add_naive_forecast(results, seasonality=7)

# Calculate metrics
metrics = calculate_metrics(results)

# Create summary and detailed tables
summary_df, detailed_df = create_results_tables(results, metrics)

# Display summary metrics
print("\n====== Model Performance Summary ======")
print(summary_df)

# Display sample of detailed results for first series
print("\n====== Sample Detailed Results (First Series) ======")
print(detailed_df[detailed_df['series_id'] == detailed_df['series_id'].iloc[0]].head(10))

# Save results to CSV if needed
summary_df.to_csv('model_summary_metrics.csv')
detailed_df.to_csv('detailed_predictions.csv', index=False)

# Find the top 4 series_id with the highest average sales
average_sales = {series_id: np.mean(detailed_df['actuals']) for series_id, detailed_df in results.items()}
high_sales = sorted(average_sales.items(), key=lambda x: x[1], reverse=True)

# print("\nTop 4 Series with Highest Average Sales:")
# for series_id, avg_sales in high_sales:
#     print(f"Series {series_id}: Average Sales = {avg_sales:.4f}")

# Plot a single series
plot_series_predictions(detailed_df, series_id=10)
plt.show()

# Plot multiple series
series_to_plot = [31, 18, 37]  # Choose any series IDs you want
plot_multiple_series(detailed_df, series_to_plot)
plt.show()

############################


def select_interesting_series(results, metrics_df, n=3):
    """
    Select interesting series based on different criteria.
    
    Args:
        results: Dictionary containing predictions and actuals
        metrics_df: DataFrame containing detailed metrics
        n: Number of series to select
        
    Returns:
        List of selected series IDs
    """
    selected_ids = []
    
    # 1. Series with highest volatility
    volatilities = {
        series_id: np.std(data['actuals']) 
        for series_id, data in results.items()
    }
    most_volatile = max(volatilities.items(), key=lambda x: x[1])[0]
    selected_ids.append(most_volatile)
    
    # 2. Series with largest prediction divergence
    divergences = {}
    for series_id, data in results.items():
        model_preds = np.array([preds for preds in data['predictions'].values()])
        divergence = np.mean(np.std(model_preds, axis=0))
        divergences[series_id] = divergence
    most_divergent = max(divergences.items(), key=lambda x: x[1])[0]
    if most_divergent not in selected_ids:
        selected_ids.append(most_divergent)
    
    # 3. Series with median volatility
    sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1])
    median_volatile = sorted_volatilities[len(sorted_volatilities)//2][0]
    if median_volatile not in selected_ids:
        selected_ids.append(median_volatile)
    
    return selected_ids[:n]

# Select interesting series to plot
selected_series = select_interesting_series(results, detailed_df)


# Print some context about the selected series
print("\nSelected Series Information:")
for series_id in selected_series:
    volatility = np.std([v for v in results[series_id]['actuals']])
    mean_value = np.mean([v for v in results[series_id]['actuals']])
    print(f"\nSeries {series_id}:")
    print(f"Mean value: {mean_value:.4f}")
    print(f"Volatility: {volatility:.4f}")