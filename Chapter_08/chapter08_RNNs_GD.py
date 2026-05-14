# =============================================================================
# Chapter 8: Recurrent Neural Networks
# =============================================================================
# Code companion for Chapter 8
# Covers: simple RNN cell mechanics, RNN state dynamics, LSTM intuition,
#         building an RNN forecaster with PyTorch Lightning, sinusoidal toy
#         dataset, naive last-observation baseline comparison
# =============================================================================

import os
import platform
import dataclasses
from functools import cached_property
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
    data_folder = Path.cwd()  / "Data" # / "Time-Series-with-PyTorch"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})

_num_workers = 0 if platform.system() == 'Windows' else min(os.cpu_count() or 1, 4)


# =============================================================================
# 8.1  Simple RNN Cell (conceptual — from the book's display code)
# =============================================================================

class RNNState:
    """Describes a RNN state and computes the history of the states
    based on inputs.

    :param w_h: W_h, the weight for h
    :param w_i: W_i, the weight for x
    :param b: b, the bias before applying activation
    :param h_init: the initial hidden state
    :param activation: the activation function to be applied.
    """

    def __init__(
        self,
        w_h: npt.ArrayLike,
        w_i: npt.ArrayLike,
        b: npt.ArrayLike,
        h_init: npt.ArrayLike,
        activation: callable = np.tanh,
    ):
        self.w_h = np.array(w_h)
        self.w_i = np.array(w_i)
        self.b = np.array(b)
        self.h_init = np.array(h_init)
        self.activation = activation

    def compute_new_state(
        self, h_current: npt.ArrayLike, z_i: npt.ArrayLike
    ) -> tuple:
        h_new = self.activation(
            np.dot(self.w_h, h_current)
            + np.dot(self.w_i, z_i) + self.b
        )
        h_delta = h_new - h_current
        return h_new, h_delta

    @cached_property
    def _metadata(self) -> dict:
        return {
            "experiment": (
                f"w_h={np.array2string(self.w_h)};w_i={np.array2string(self.w_i)};"
                f"b={np.array2string(self.b)};h_init={np.array2string(self.h_init)};"
                f"{self.activation.__name__}"
            ),
            "w_h": np.array2string(self.w_h),
            "w_i": np.array2string(self.w_i),
            "b": np.array2string(self.b),
            "activation": self.activation.__name__,
            "initial_state": np.array2string(self.h_init),
        }

    def states(self, z: npt.ArrayLike) -> dict:
        h_t = [self.h_init]
        t_steps = [0]
        h_t_delta = [np.zeros_like(self.h_init)]
        for t, z_i in enumerate(z):
            h_new, h_delta = self.compute_new_state(
                h_current=h_t[-1], z_i=z_i
            )
            h_t.append(h_new)
            h_t_delta.append(h_delta)
            t_steps.append(t + 1)

        total_time_steps = len(t_steps)

        return {
            **{
                "t": np.array(t_steps),
                "h": np.array(h_t),
                "dh": np.array(h_t_delta),
                "z": np.pad(z, (1, 0), constant_values=0)
            },
            **{
                k: [v] * total_time_steps
                for k, v in self._metadata.items()
            },
        }


# --- Demonstrate RNN cell dynamics with different parameters ---
scenarios = [
    {"w_h": 0.5, "w_i": 1, "b": 0, "h_init": 0.1, "label": "w_h=0.5, h0=0.1"},
    {"w_h": 1.5, "w_i": 1, "b": 0, "h_init": 0.1, "label": "w_h=1.5, h0=0.1"},
    {"w_h": 0.5, "w_i": 1, "b": 0, "h_init": 2.0, "label": "w_h=0.5, h0=2.0"},
    {"w_h": 1.5, "w_i": 1, "b": 0, "h_init": 2.0, "label": "w_h=1.5, h0=2.0"},
]

t = np.linspace(0, 20, 200)
z_linear = 0.1 * t
z_sin = np.sin(t)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for input_data, input_name, ax in zip(
    [z_linear, z_sin], ["Linear Input", "Sin Input"], axes
):
    ax.plot(t, input_data, 'k--', alpha=0.4, label='Input z')
    for i, s in enumerate(scenarios):
        rnn = RNNState(w_h=s["w_h"], w_i=s["w_i"], b=s["b"], h_init=s["h_init"])
        result = rnn.states(input_data)
        ax.plot(result["t"], result["h"],
                color=custom_palette[(i + 1) % len(custom_palette)],
                label=s["label"])
    ax.set_title(f"Simple RNN: {input_name}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Hidden state h")
    ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


# =============================================================================
# 8.2  Toy Dataset — Sinusoid
# =============================================================================

df = pd.DataFrame({
    "t": np.linspace(0, 100, 501),
    "y": np.sin(np.linspace(0, 100, 501))
})

# --- Figure 8.X: Sinusoidal toy dataset ---
_, ax = plt.subplots(figsize=(10, 5))
df.plot(x="t", y="y", ax=ax, color=custom_palette[1])
ax.set_title("Toy Sinusoidal Dataset")
ax.set_xlabel("t")
ax.set_ylabel("y")
plt.show()


# =============================================================================
# 8.3  RNN Model Definition
# =============================================================================

@dataclasses.dataclass
class TSRNNParams:
    """Parameters for the RNN model."""
    input_size: int
    hidden_size: int
    num_layers: int = 1


class TSRNN(nn.Module):
    """RNN for univariate time series modelling."""

    def __init__(self, history_length: int, horizon: int,
                 rnn_params: TSRNNParams):
        super().__init__()
        self.rnn_params = rnn_params
        self.history_length = history_length
        self.horizon = horizon

        self.regulate_input = nn.Linear(
            self.history_length, self.rnn_params.input_size)
        self.rnn = nn.RNN(
            input_size=self.rnn_params.input_size,
            hidden_size=self.rnn_params.hidden_size,
            num_layers=self.rnn_params.num_layers,
            batch_first=True)
        self.regulate_output = nn.Linear(
            self.rnn_params.hidden_size, self.horizon)

    @property
    def rnn_config(self):
        return dataclasses.asdict(self.rnn_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regulate_input(x)
        x, _ = self.rnn(x)
        return self.regulate_output(x)


# =============================================================================
# 8.4  Lightning Forecaster
# =============================================================================

class RNNForecaster(L.LightningModule):
    """Forecaster based on RNN."""

    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.rnn(x)
        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.rnn(x)
        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)
        y_hat = self.rnn(x)
        return x, y_hat

    def forward(self, x: torch.Tensor):
        x = x.squeeze().type(self.dtype)
        return x, self.rnn(x)


# =============================================================================
# 8.5  Data Module (ts_bolt — pip install ts_bolt)
# =============================================================================

from ts_bolt.datamodules.pandas import DataFrameDataModule


# =============================================================================
# 8.6  Instantiate and Train RNN
# =============================================================================

history_length_1_step = 100
horizon_1_step = 1

pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]],
)

from torchinfo import summary

ts_rnn_params_1_step = TSRNNParams(
    input_size=100,
    hidden_size=64,
    num_layers=1
)

ts_rnn_1_step = TSRNN(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    rnn_params=ts_rnn_params_1_step,
)

summary(ts_rnn_1_step)

rnn_forecaster_1_step = RNNForecaster(rnn=ts_rnn_1_step)

trainer_1_step = L.Trainer(
    precision="64",
    max_epochs=100,
    min_epochs=5,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min",
            min_delta=1e-7, patience=3)
    ],
    logger=False,
    enable_checkpointing=False,
)

trainer_1_step.fit(model=rnn_forecaster_1_step, datamodule=pdm_1_step)


# =============================================================================
# 8.7  Predict and Compare with Naive Forecaster
# =============================================================================

predictions_1_step = trainer_1_step.predict(
    model=rnn_forecaster_1_step, datamodule=pdm_1_step)


# Naive last-observation forecaster
from ts_bolt.naive_forecasters.last_observation import LastObservationForecaster


trainer_naive_1_step = L.Trainer(
    precision="64", logger=False, enable_checkpointing=False)

lobs_forecaster_1_step = LastObservationForecaster(horizon=horizon_1_step)
lobs_1_step_predictions = trainer_naive_1_step.predict(
    model=lobs_forecaster_1_step, datamodule=pdm_1_step)


# --- Figure: Comparing RNN with Naive Forecaster ---
from ts_bolt.evaluation.evaluator import Evaluator

evaluator_1_step = Evaluator(step=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
        color=custom_palette[3], label="Truth")
ax.plot(evaluator_1_step.y(predictions_1_step),
        color=custom_palette[2], linestyle='--', label="RNN Predictions")
ax.plot(evaluator_1_step.y(lobs_1_step_predictions),
        color=custom_palette[1], linestyle='-.', label="Naive Predictions")
ax.set_xlabel("Sample")
ax.set_ylabel("y")
ax.set_title("Comparing RNN with Naive Forecaster")
ax.legend()
plt.tight_layout()
plt.show()

# --- Metrics comparison ---
metrics_comparison = pd.merge(
    evaluator_1_step.metrics(predictions_1_step, pdm_1_step.predict_dataloader()),
    evaluator_1_step.metrics(lobs_1_step_predictions, pdm_1_step.predict_dataloader()),
    how="inner",
    left_index=True, right_index=True,
    suffixes=["_rnn", "_last_obs"]
)
print(metrics_comparison.to_string())