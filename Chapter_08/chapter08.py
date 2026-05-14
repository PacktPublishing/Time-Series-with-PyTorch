# Extracted from chapter8.qmd
# Do not edit the source .qmd file directly.

#| label: fig-rnn-forecast-toy-sin
#| fig-cap: "Toy dataset generated using sinusoid."
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(
    {
        "t": np.linspace(0, 100, 501),
        "y": np.sin(np.linspace(0, 100, 501))
    }
)

_, ax = plt.subplots(figsize=(5, 3.09))

df.plot(x="t", y="y", ax=ax);

# ----------------------------------------------------------------------

#| results: false
import dataclasses
import torch
from torch import nn

@dataclasses.dataclass
class TSRNNParams:
    """A dataclass to be served as our parameters
    for the RNN model.

    :param hidden_size: number of dimensions 
        in the hidden state
    :param input_size: input dim
    :param num_layers: number of units stacked
    """

    input_size: int
    hidden_size: int
    num_layers: int = 1


class TSRNN(nn.Module):
    """RNN for univaraite time series modeling.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param rnn_params: the parameters for the RNN network.
    """

    def __init__(
            self, history_length: int, 
            horizon: int, 
            rnn_params: TSRNNParams
        ):
        super().__init__()
        self.rnn_params = rnn_params
        self.history_length = history_length
        self.horizon = horizon

        self.regulate_input = nn.Linear(
            self.history_length,
            self.rnn_params.input_size
        )

        self.rnn = nn.RNN(
            input_size=self.rnn_params.input_size,
            hidden_size=self.rnn_params.hidden_size,
            num_layers=self.rnn_params.num_layers,
            batch_first=True,
        )

        self.regulate_output = nn.Linear(
            self.rnn_params.hidden_size,
            self.horizon
        )

    @property
    def rnn_config(self):
        return dataclasses.asdict(self.rnn_params)

    def forward(
            self, x: torch.Tensor
        ) -> torch.Tensor:
        x = self.regulate_input(x)
        x, _ = self.rnn(x)

        return self.regulate_output(x)

# ----------------------------------------------------------------------

#| results: false
history_length_1_step = 100
horizon_1_step = 1

# ----------------------------------------------------------------------

#| results: false
import lightning as L


class RNNForecaster(L.LightningModule):
    """Forecaster based on RNN

    :param rnn: RNN model that takes time 
        series input and predicts time series
        of the specified horizon.
    """
    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3
        )
        return optimizer

    def training_step(
            self, 
            batch: tuple[torch.Tensor], 
            batch_idx: int
        ) -> torch.Tensor:
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)

        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True
        )
        return loss

    def validation_step(
            self, 
            batch: tuple[torch.Tensor], 
            batch_idx: int
        ) -> torch.Tensor:
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)

        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict(
            {"val_loss": loss},
            prog_bar=True
        )
        return loss

    def predict_step(
            self, 
            batch: tuple[torch.Tensor], 
            batch_idx: int
        ) -> tuple[torch.Tensor]:
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)
        return x, y_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = x.squeeze().type(self.dtype)
        return x, self.rnn(x)

# ----------------------------------------------------------------------

#| results: false
from ts_bolt.datamodules.pandas import DataFrameDataModule


pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]],
)

# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------

rnn_forecaster_1_step = RNNForecaster(
    rnn=ts_rnn_1_step
)

# ----------------------------------------------------------------------

#| results: false
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


logger_1_step = L.pytorch.loggers.TensorBoardLogger(
    save_dir="lightning_logs",
    name="rnn_ts_1_step"
)

trainer_1_step = L.Trainer(
    precision="64",
    max_epochs=100,
    min_epochs=5,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=1e-7,
            patience=3
        )
    ],
    logger=logger_1_step,
)

# ----------------------------------------------------------------------

#| results: false
trainer_1_step.fit(
    model=rnn_forecaster_1_step,
    datamodule=pdm_1_step
)

# ----------------------------------------------------------------------

#| results: false
predictions_1_step = trainer_1_step.predict(
    model=rnn_forecaster_1_step, datamodule=pdm_1_step
)

# ----------------------------------------------------------------------

#| results: false
from ts_bolt.naive_forecasters.last_observation import LastObservationForecaster


trainer_naive_1_step = L.Trainer(precision="64")

lobs_forecaster_1_step = LastObservationForecaster(horizon=horizon_1_step)
lobs_1_step_predictions = trainer_naive_1_step.predict(
    model=lobs_forecaster_1_step, datamodule=pdm_1_step
)

# ----------------------------------------------------------------------

#| label: fig-rnn-forecast-toy-compare-naive
#| fig-cap: "Comparing RNN with Naive Forecaster."
from ts_bolt.evaluation.evaluator import Evaluator


evaluator_1_step = Evaluator(step=0)

fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
    "g-",
    label="truth",
)

ax.plot(evaluator_1_step.y(predictions_1_step), "r--", label="predictions")

ax.plot(evaluator_1_step.y(lobs_1_step_predictions), "b-.", label="naive predictions")

plt.legend()

# ----------------------------------------------------------------------

pd.merge(
    evaluator_1_step.metrics(predictions_1_step, pdm_1_step.predict_dataloader()),
    evaluator_1_step.metrics(lobs_1_step_predictions, pdm_1_step.predict_dataloader()),
    how="inner",
    left_index=True, right_index=True,
    suffixes=["_rnn", "_last_obs"]
)