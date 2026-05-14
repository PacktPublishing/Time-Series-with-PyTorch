# =============================================================================
# Chapter 9: Transformers
# =============================================================================
# Code companion for Chapter 9
# Covers: attention mechanism, positional encoding, vanilla transformer for
#         time series, PyTorch implementation, NeuralForecast transformer
#         models on M5 data (ARIMA baseline, VanillaTransformer, iTransformer)
# =============================================================================

import os
import platform
import math
import dataclasses
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ts_bolt.datamodules.pandas import DataFrameDataModule
from ts_bolt.naive_forecasters.last_observation import LastObservationForecaster
from ts_bolt.evaluation.evaluator import Evaluator

import matplotlib.pyplot as plt

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
    data_folder = Path.cwd() / "Time-Series-with-PyTorch" / "Data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})

_num_workers = 0 if platform.system() == 'Windows' else min(os.cpu_count() or 1, 4)


# =============================================================================
# 9.1  Positional Encoding (numpy)
# =============================================================================

class PositionalEncodingSimple:
    """Positional encoding written in numpy.

    :param d_model: hidden dimension of the encoder
    :param max_len: maximum length of our positional encoder.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        position = np.expand_dims(np.arange(max_len), axis=1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.pe = np.zeros((max_len, d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pe[:x.shape[1]]


# --- Figure: Positional encoding heatmap ---
pes = PositionalEncodingSimple(d_model=50)
x_pes_in = np.ones((1, 10, 1))
x_pes_out = pes(x=x_pes_in)

_, ax = plt.subplots(figsize=(10, 6))
ax.matshow(x_pes_out, cmap="cividis")
ax.set_xlabel("Embedding")
ax.set_ylabel("Temporal")
ax.set_title("Positional Encoding")
plt.show()


# =============================================================================
# 9.2  Positional Encoding (PyTorch)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding to be added to input embedding.

    :param d_model: hidden dimension of the encoder
    :param dropout: rate of dropout
    :param max_len: maximum length of positional encoder.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history_length = x.size(1)
        x = x + self.pe[:history_length]
        return self.dropout(x)


# =============================================================================
# 9.3  Transformer Model
# =============================================================================

@dataclasses.dataclass
class TSTransformerParams:
    """Hyperparameters for the transformer model."""
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dropout: int = 0.1


class TSTransformer(nn.Module):
    """Transformer for univariate time series forecasting.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param transformer_params: all the parameters.
    """

    def __init__(self, history_length: int, horizon: int,
                 transformer_params: TSTransformerParams):
        super().__init__()
        self.transformer_params = transformer_params
        self.history_length = history_length
        self.horizon = horizon

        self.embedding = nn.Linear(1, self.transformer_params.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_params.d_model,
            nhead=self.transformer_params.nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_params.num_encoder_layers
        )

        self.reverse_embedding = nn.Linear(
            self.transformer_params.d_model, 1)
        self.decoder = nn.Linear(self.history_length, self.horizon)

    @property
    def transformer_config(self) -> dict:
        return dataclasses.asdict(self.transformer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        encoder_state = self.encoder(x)
        decoder_in = self.reverse_embedding(encoder_state).squeeze(-1)
        return self.decoder(decoder_in)


# =============================================================================
# 9.4  Lightning Forecaster
# =============================================================================

class TransformerForecaster(L.LightningModule):
    """Transformer forecasting: training, validation, prediction."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.transformer(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.transformer(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.transformer(x)
        return x, y_hat

    def forward(self, x: torch.Tensor):
        return x, self.transformer(x)


# =============================================================================
# 9.5  Toy Dataset and Training
# =============================================================================

df = pd.DataFrame({
    "t": np.linspace(0, 100, 501),
    "y": np.sin(np.linspace(0, 100, 501))
})

# --- Figure: Sinusoidal dataset ---
_, ax = plt.subplots(figsize=(10, 5))
df.plot(x="t", y="y", ax=ax, color=custom_palette[1])
ax.set_title("Toy Sinusoidal Dataset")
plt.show()


# Hyperparameters
history_length_1_step = 100
horizon_1_step = 1

ts_transformer_params_1_step = TSTransformerParams(
    d_model=192, nhead=6, num_encoder_layers=1
)
print(ts_transformer_params_1_step)

# Instantiate model
ts_transformer_1_step = TSTransformer(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    transformer_params=ts_transformer_params_1_step,
)
print(ts_transformer_1_step)

# Wrap in forecaster
transformer_forecaster_1_step = TransformerForecaster(
    transformer=ts_transformer_1_step)

# Data module (ts_bolt)
pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]],
)

# Train
trainer_1_step = L.Trainer(
    precision="64",
    max_epochs=1000, # was 100 in chapter, increased for better convergence
    min_epochs=5,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min",
            min_delta=1e-7, patience=3)
    ],
    logger=False,
    enable_checkpointing=False,
)

trainer_1_step.fit(
    model=transformer_forecaster_1_step,
    datamodule=pdm_1_step
)


# =============================================================================
# 9.6  Hyperparameter Tuning with Optuna
# =============================================================================
# The default hyperparameters (d_model=192, nhead=6, SGD lr=1e-3) were chosen
# for the book's architecture walkthrough, not for performance. On a simple
# sinusoid, a transformer can easily underperform a naive baseline without
# tuning. Here we use Optuna to find reasonable hyperparameters.

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    d_model_choice = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # nhead must divide d_model
    if d_model_choice % nhead != 0:
        raise optuna.TrialPruned()

    params = TSTransformerParams(
        d_model=d_model_choice,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers
    )

    torch.manual_seed(42)
    model = TSTransformer(
        history_length=history_length_1_step,
        horizon=horizon_1_step,
        transformer_params=params
    )

    class TransformerForecasterTuning(L.LightningModule):
        def __init__(self, transformer, lr):
            super().__init__()
            self.transformer = transformer
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y = y.squeeze(-1).type(self.dtype)
            loss = nn.functional.mse_loss(self.transformer(x), y)
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y = y.squeeze(-1).type(self.dtype)
            loss = nn.functional.mse_loss(self.transformer(x), y)
            self.log("val_loss", loss, prog_bar=False)
            return loss

    forecaster = TransformerForecasterTuning(model, lr)

    trainer = L.Trainer(
        precision="64",
        max_epochs=200,
        min_epochs=10,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", mode="min",
                min_delta=1e-7, patience=5)
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model=forecaster, datamodule=pdm_1_step)

    return trainer.callback_metrics.get("val_loss", float("inf")).item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print(f"\nBest trial val_loss: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")


# =============================================================================
# 9.7  Train Best Model and Compare with Naive
# =============================================================================

best = study.best_params
best_params = TSTransformerParams(
    d_model=best['d_model'],
    nhead=best['nhead'],
    num_encoder_layers=best['num_encoder_layers']
)

torch.manual_seed(42)
best_transformer = TSTransformer(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    transformer_params=best_params
)

# Reuse the original TransformerForecaster but swap optimizer to Adam
class TransformerForecasterTuned(L.LightningModule):
    def __init__(self, transformer, lr):
        super().__init__()
        self.transformer = transformer
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        loss = nn.functional.mse_loss(self.transformer(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        loss = nn.functional.mse_loss(self.transformer(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)
        return x, self.transformer(x)

    def forward(self, x):
        return x, self.transformer(x)


best_forecaster = TransformerForecasterTuned(best_transformer, best['lr'])

trainer_best = L.Trainer(
    precision="64",
    max_epochs=200,
    min_epochs=10,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min",
            min_delta=1e-7, patience=10)
    ],
    logger=False,
    enable_checkpointing=False,
)
trainer_best.fit(model=best_forecaster, datamodule=pdm_1_step)

# Predict
predictions_best = trainer_best.predict(
    model=best_forecaster, datamodule=pdm_1_step)

# Naive baseline
trainer_naive = L.Trainer(
    precision="64", logger=False, enable_checkpointing=False)
lobs_forecaster = LastObservationForecaster(horizon=horizon_1_step)
lobs_predictions = trainer_naive.predict(
    model=lobs_forecaster, datamodule=pdm_1_step)

# --- Figure: Comparing Tuned Transformer with Naive Forecaster ---
evaluator_1_step = Evaluator(step=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
    color=custom_palette[3], label="Truth")
ax.plot(
    evaluator_1_step.y(predictions_best),
    color=custom_palette[2], linestyle='--', label="Transformer (tuned)")
ax.plot(
    evaluator_1_step.y(lobs_predictions),
    color=custom_palette[1], linestyle='-.', label="Naive Predictions")
ax.set_xlabel("Sample")
ax.set_ylabel("y")
ax.set_title("Tuned Transformer vs Naive Forecaster")
ax.legend()
plt.tight_layout()
plt.show()

# Metrics comparison
metrics_comparison = pd.merge(
    evaluator_1_step.metrics(predictions_best, pdm_1_step.predict_dataloader()),
    evaluator_1_step.metrics(lobs_predictions, pdm_1_step.predict_dataloader()),
    how="inner",
    left_index=True, right_index=True,
    suffixes=["_transformer", "_last_obs"]
)
print(metrics_comparison.to_string())


# =============================================================================
# 9.8  Real-World Example: M5 Data with NeuralForecast (not run in chapter, 
# but feel free to run if you have the resources!)
# =============================================================================

# import seaborn as sns
# from statsmodels.graphics.tsaplots import plot_acf
# from statsforecast import StatsForecast
# from statsforecast.models import AutoARIMA
# from neuralforecast import NeuralForecast
# from neuralforecast.models import VanillaTransformer, iTransformer
#
# # Load M5 California aggregate data
# data_source = (
#     "https://raw.githubusercontent.com/datumorphism/"
#     "dataset-m5-simplified/b486cd6a3e183b80016a91ba0fd9b19493cdc587/"
#     "dataset/m5_store_sales.csv"
# )
# df_m5 = (
#     pd.read_csv(data_source, parse_dates=["date"])
#     .rename(columns={"date": "ds", "CA": "y"})[["ds", "y"]]
#     .assign(unique_id=1)
# )
#
# train_test_split_date = "2016-01-01"
# df_train = df_m5[df_m5.ds <= train_test_split_date]
# df_test = df_m5[df_m5.ds > train_test_split_date]
# horizon = 3
#
# # ARIMA baseline
# sf = StatsForecast(
#     models=[AutoARIMA(seasonal=False)],
#     freq='d'
# )
#
# def forecast_test(forecaster_pred, df_train, df_test):
#     dfs_pred = []
#     for i in range(len(df_test)):
#         df_pred_input_i = pd.concat(
#             [df_train, df_test[:i]]
#         ).reset_index(drop=True)
#         df_pred_output_i = forecaster_pred(df_pred_input_i)
#         df_pred_output_i["step"] = i
#         dfs_pred.append(df_pred_output_i)
#     return pd.concat(dfs_pred)
#
# df_y_hat_arima = forecast_test(
#     lambda x: sf.forecast(df=x, h=horizon),
#     df_train=df_train, df_test=df_test
# )
#
# # Transformer models
# models = [
#     VanillaTransformer(
#         hidden_size=128, n_head=4, learning_rate=0.0001,
#         scaler_type="robust", max_steps=500, batch_size=32,
#         windows_batch_size=512, random_seed=16,
#         input_size=30, step_size=3, h=horizon,
#     ),
#     iTransformer(
#         input_size=30, h=horizon, max_steps=50, n_series=1,
#     )
# ]
# nf = NeuralForecast(models=models, freq="d")
# nf.fit(df=df_train)
#
# df_y_hat_transformer = forecast_test(
#     nf.predict, df_train=df_train, df_test=df_test
# )