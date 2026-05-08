import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import TimeSeriesDataset

"""
During training, PyTorch Lightning automatically saved your model checkpoints in the './checkpoints' directory. These checkpoints contain:

The model's architecture
The trained weights
Training state information
Optimization parameters


To load the model, we need to:

First create a new instance of the model with the same architecture
Then load the saved weights from the checkpoint file
Put the model in evaluation mode for making predictions


The checkpoints are typically named with a pattern like:

'concat-best.ckpt' (if using the best validation performance)
Or with epoch numbers and metrics in the filename


After loading, the model will have exactly the same weights and behavior as it did at the point when the checkpoint was saved.

Important things to note:

Make sure you're using the same model architecture parameters when loading as you used during training
The number of series (num_series parameter) must match what you used during training
The checkpoint file path needs to point to where your model was actually saved
"""


def setup_module_imports():
    """Set up the Python path to find our modules."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    module_dir = os.path.join(project_root, 'example_notebooks', 'chp11_notebooks', 'simple_gfm_pytorch')
    
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    return module_dir

# Set up imports
module_dir = setup_module_imports()

# Import our modules
from train import GlobalForecastingModel, load_m5_data
from dataset import TimeSeriesDataset

def worker_init_fn(worker_id):
    """Initialize each worker process with the correct path."""
    if module_dir not in sys.path:
        sys.path.append(module_dir)

def create_data_loader(dataset, batch_size=32, num_workers=4):
    """Create a DataLoader with proper worker initialization."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

def load_trained_model(checkpoint_path, num_series):
    """
    Load a trained model from checkpoint with proper configuration.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_series (int): Number of series in the dataset
        
    Returns:
        model: Loaded PyTorch Lightning model in evaluation mode
    """
    # Load the checkpoint to get the hyperparameters
    checkpoint = torch.load(checkpoint_path)
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create model with saved hyperparameters, falling back to defaults if needed
    model = GlobalForecastingModel(
        num_series=num_series,  # Must match current dataset
        input_dim=hparams.get('input_dim', 2),
        hidden_dim=hparams.get('hidden_dim', 32),
        num_layers=hparams.get('num_layers', 1),
        dropout=hparams.get('dropout', 0.1),
        learning_rate=hparams.get('learning_rate', 1e-3)
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def evaluate_model(model, loader, device='cuda'):
    """
    Compute detailed evaluation metrics on the given loader.
    Returns both overall and per-series metrics.
    
    Args:
        model: The trained forecasting model
        loader: DataLoader containing validation/test data
        device: Device to run computations on ('cuda' or 'cpu')
        
    Returns:
        dict: Contains 'overall' and 'per_series' metrics including MSE, RMSE, MAE, MAPE
    """
    model.eval()
    model = model.to(device)
    
    overall_metrics = {'mse': [], 'mae': [], 'mape': []}
    per_series_metrics = {}
    
    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            sid = batch['series_idx'].to(device)
            
            # Generate predictions
            preds = model(x, sid).squeeze(-1)
            
            # Calculate metrics
            mse = F.mse_loss(preds, y[:, 0], reduction='none')
            mae = F.l1_loss(preds, y[:, 0], reduction='none')
            
            # Calculate MAPE avoiding division by zero
            epsilon = 1e-7
            mape = torch.abs((preds - y[:, 0]) / (y[:, 0] + epsilon)) * 100
            
            # Store metrics
            overall_metrics['mse'].extend(mse.cpu().numpy())
            overall_metrics['mae'].extend(mae.cpu().numpy())
            overall_metrics['mape'].extend(mape.cpu().numpy())
            
            # Store per-series metrics
            for i, series_id in enumerate(sid.cpu().numpy()):
                if series_id[0] not in per_series_metrics:
                    per_series_metrics[series_id[0]] = {'mse': [], 'mae': [], 'mape': []}
                per_series_metrics[series_id[0]]['mse'].append(mse[i].item())
                per_series_metrics[series_id[0]]['mae'].append(mae[i].item())
                per_series_metrics[series_id[0]]['mape'].append(mape[i].item())
    
    # Calculate final metrics
    results = {
        'overall': {
            'mse': np.mean(overall_metrics['mse']),
            'rmse': np.sqrt(np.mean(overall_metrics['mse'])),
            'mae': np.mean(overall_metrics['mae']),
            'mape': np.mean(overall_metrics['mape'])
        },
        'per_series': {
            sid: {
                'mse': np.mean(metrics['mse']),
                'rmse': np.sqrt(np.mean(metrics['mse'])),
                'mae': np.mean(metrics['mae']),
                'mape': np.mean(metrics['mape'])
            }
            for sid, metrics in per_series_metrics.items()
        }
    }
    
    return results

def generate_predictions(model, loader, device='cuda', max_series=None):
    """
    Generate predictions for all series in the loader.
    
    Args:
        model: The trained forecasting model
        loader: DataLoader containing validation/test data
        device: Device to run computations on ('cuda' or 'cpu')
        max_series: Maximum number of series to process (None for all)
        
    Returns:
        dict: Dictionary containing actual values and predictions for each series
    """
    model.eval()
    model = model.to(device)
    predictions = {}
    
    with torch.no_grad():
        series_count = 0
        for batch in loader:
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            sid = batch['series_idx'].to(device)
            
            # Generate predictions
            preds = model(x, sid).squeeze(-1)
            
            # Store results
            for i, series_id in enumerate(sid.cpu().numpy()):
                series_id = series_id[0]
                if series_id not in predictions:
                    predictions[series_id] = {
                        'actuals': [],
                        'predictions': [],
                        'timestamps': []
                    }
                    series_count += 1
                
                predictions[series_id]['actuals'].append(y[i, 0].cpu().item())
                predictions[series_id]['predictions'].append(preds[i].cpu().item())
                
                if max_series and series_count >= max_series:
                    return predictions
    
    return predictions

def plot_predictions(predictions, idx_to_series, num_plots=4, figsize=(15, 10)):
    """
    Plot actual vs predicted values for multiple series.
    
    Args:
        predictions: Dictionary containing predictions and actual values
        idx_to_series: Mapping from series indices to series names
        num_plots: Number of series to plot
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    num_series = min(len(predictions), num_plots)
    fig, axes = plt.subplots(num_series, 1, figsize=figsize)
    if num_series == 1:
        axes = [axes]
    
    for i, (series_id, data) in enumerate(list(predictions.items())[:num_series]):
        series_name = idx_to_series[series_id]
        actuals = data['actuals']
        preds = data['predictions']
        
        axes[i].plot(actuals, label='Actual', marker='o')
        axes[i].plot(preds, label='Predicted', marker='x')
        axes[i].set_title(f'Series {series_name}')
        axes[i].legend()
        axes[i].grid(True)
        
        # Calculate metrics for this series
        mse = np.mean((np.array(actuals) - np.array(preds))**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
        
        axes[i].text(0.02, 0.98, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}',
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

