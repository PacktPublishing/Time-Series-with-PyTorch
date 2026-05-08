#train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import TimeSeriesDataset
import time

# Data preparation function
def load_m5_data(path: str):
    """Load and prepare M5 data, removing the last 28 days."""
    print("Loading data...")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove last 28 days
    cutoff = df['date'].max() - pd.Timedelta(days=28)
    df = df[df['date'] <= cutoff]
    
    # Split based on date (80/20)
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=int(0.2 * (max_date - df['date'].min()).days))
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    
    # Convert to dictionary format
    def prepare_data(df):
        df = df.sort_values(['date', 'id'])
        series_dict = {}
        for sid in df['id'].unique():
            sub = df[df['id'] == sid].sort_values('date')
            feats = np.column_stack([sub['sold'].values, sub['sell_price'].values])
            if len(feats) > 0:
                series_dict[sid] = feats
        return series_dict
    
    return prepare_data(train_df), prepare_data(test_df)

# Model Architecture
class TimeSeriesArchitecture(nn.Module):
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
            hidden_dim, hidden_dim, num_layers=num_layers,
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
        series_embed = self.series_embeddings(series_ids.squeeze())
        combined = torch.cat([temporal_out[:, -1, :], series_embed], dim=1)
        return self.decoder(combined)

# Lightning Module
class GlobalForecastingModel(L.LightningModule):
    def __init__(self, num_series, input_dim=2, hidden_dim=64, num_layers=2, 
                 dropout=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TimeSeriesArchitecture(
            num_series=num_series,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.learning_rate = learning_rate
    
    def forward(self, x, series_ids):
        return self.model(x, series_ids)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        sid = batch['series_idx']
        y_hat = self(x, sid).squeeze(-1)
        loss = F.mse_loss(y_hat, y[:, 0])
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        sid = batch['series_idx']
        y_hat = self(x, sid).squeeze(-1)
        val_loss = F.mse_loss(y_hat, y[:, 0])
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# Progress callback
class SimpleProgress(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch + 1}")
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds")

def main():
    # Set the path to your data
    m5_csv = "C:/Users/retailexp/Documents/GitHub/tsfwpt/data/M5_t20_ABC.csv"
    
    # Load data
    train_dict, test_dict = load_m5_data(m5_csv)
    
    # Create optimized dataloaders
    train_dataset = TimeSeriesDataset(train_dict, lookback=14, horizon=1)
    test_dataset = TimeSeriesDataset(test_dict, lookback=14, horizon=1)
    
    # Create dataloaders with multiple workers
    loader_config = {
        'batch_size': 32,
        'num_workers': 4,  # Adjust based on your CPU
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'shuffle': False
    }
    
    train_loader = DataLoader(train_dataset, **loader_config)
    val_loader = DataLoader(test_dataset, **loader_config)
    
    # Initialize model
    model = GlobalForecastingModel(
        num_series=len(train_dict),
        input_dim=2,
        hidden_dim=32,
        num_layers=1,
        dropout=0.1,
        learning_rate=1e-3
    )
    
    # Configure trainer
    trainer = L.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks=[SimpleProgress()],
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()