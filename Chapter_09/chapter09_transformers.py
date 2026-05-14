# Extracted from chapter9_transformers.qmd
# Do not edit the source .qmd file directly.

import math
import numpy as np


class PositionalEncodingSimple:
    """Positional encoding for our transformer
    written in numpy.

    :param d_model: hidden dimension of the encoder
    :param max_len: maximum length of our positional
        encoder. The encoder can not encode sequence
        length longer than max_len.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        position = np.expand_dims(
            np.arange(max_len), axis=1
        )
        div_term = np.exp(
            np.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )
        self.pe = np.zeros((max_len, d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: input to be encoded
            with shape
            `(batch_size, sequence_length, embedding_dim)`
        """
        return self.pe[: x.shape[1]]

# ----------------------------------------------------------------------

#| label: fig-ts-transformer-position-encoding
#| fig-cap: "Positional encoding written in numpy. The color indicates the values of the positional encoding. Note that each temporal dimension (each row) is unique. When each temporal dimension is added on top of the embedded input, we annotate the positional information of each time step."
#| echo: false
import matplotlib.pyplot as plt
pes = PositionalEncodingSimple(d_model=50)
x_pes_in = np.ones((1, 10, 1))

x_pes_out = pes(
    x=x_pes_in
)

_, ax = plt.subplots(figsize=(10, 6.18))

ax.matshow(x_pes_out, cmap="cividis")
ax.set_xlabel("Embedding")
ax.set_ylabel("Temporal")
plt.show()

# ----------------------------------------------------------------------

from torch import nn
import torch


class PositionalEncoding(nn.Module):
    """Positional encoding to be added to
    input embedding.

    :param d_model: hidden dimension of the encoder
    :param dropout: rate of dropout
    :param max_len: maximum length of our positional
        encoder. The encoder can not encode sequence
        length longer than max_len.
    """

    def __init__(
            self, d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000,
        ):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input embedded time series,
            shape `[batch_size, seq_len, embedding_dim]`
        """
        history_length = x.size(1)
        x = x + self.pe[: history_length]

        return self.dropout(x)

# ----------------------------------------------------------------------

import dataclasses


@dataclasses.dataclass
class TSTransformerParams:
    """A dataclass that contains all
    the parameters for the transformer model.
    """
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dropout: int = 0.1

# ----------------------------------------------------------------------

class TSTransformer(nn.Module):
    """Transformer for univariate time series forecasting.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param transformer_params: all the parameters.
    """

    def __init__(
        self,
        history_length: int,
        horizon: int,
        transformer_params: TSTransformerParams,
    ):
        super().__init__()
        self.transformer_params = transformer_params
        self.history_length = history_length
        self.horizon = horizon

        self.embedding = nn.Linear(
            1, self.transformer_params.d_model
        )

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
            self.transformer_params.d_model, 1
        )

        self.decoder = nn.Linear(
            self.history_length, self.horizon
        )

    @property
    def transformer_config(self) -> dict:
        """all the param in dict format
        """
        return dataclasses.asdict(
            self.transformer_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input historical time series,
            shape `[batch_size, seq_len, n_var]`
        """
        x = self.embedding(x)

        encoder_state = self.encoder(x)

        decoder_in = self.reverse_embedding(
            encoder_state
        ).squeeze(-1)

        return self.decoder(decoder_in)

# ----------------------------------------------------------------------

import lightning as L


class TransformerForecaster(L.LightningModule):
    """Transformer forecasting training, validation,
    and prediction all collected in one class.

    :param transformer: pre-defined transformer model
    """
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def configure_optimizers(
            self
        )-> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3
        )

        return optimizer

    def training_step(
            self, batch: tuple[torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True
        )
        return loss

    def validation_step(
            self, batch: tuple[torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict(
            {"val_loss": loss},
            prog_bar=True
        )

        return loss

    def predict_step(
            self, batch: list[torch.Tensor],
            batch_idx: int
        )-> tuple[torch.Tensor]:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        return x, y_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:

        return x, self.transformer(x)

# ----------------------------------------------------------------------

history_length_1_step = 100
horizon_1_step = 1

# ----------------------------------------------------------------------

ts_transformer_params_1_step = TSTransformerParams(
    d_model=192, nhead=6, num_encoder_layers=1
)
ts_transformer_params_1_step

# ----------------------------------------------------------------------

ts_transformer_1_step = TSTransformer(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    transformer_params=ts_transformer_params_1_step,
)

ts_transformer_1_step

# ----------------------------------------------------------------------

transformer_forecaster_1_step = TransformerForecaster(transformer=ts_transformer_1_step)

# ----------------------------------------------------------------------

#| echo: false
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme(
    context='paper', 
    style='ticks', 
    palette='colorblind',
)


data_source = "https://raw.githubusercontent.com/datumorphism/dataset-m5-simplified/b486cd6a3e183b80016a91ba0fd9b19493cdc587/dataset/m5_store_sales.csv"

df = (
    pd
    .read_csv(data_source, parse_dates=["date"])
    .rename(columns={"date": "ds", "CA": "y"})[["ds", "y"]]
    .assign(unique_id=1)
)

df[["ds", "y"]].head(10)

# ----------------------------------------------------------------------

#| echo: false
#| label: fig-ts-transformer-data-m5-ca
#| fig-cap: "Walmart sales data extracted from the M5 dataset. In the upper panel, we have the full time series of sales in all California stores in Walmart. The middle panel shows the details of the year 2015. We observe the cyclic features of the time series. The bottom panel plots the auto-correlation of the time series. As expected, there is a 7-day cycle in sales."
_, ax = plt.subplots(figsize=(10, 13),nrows=3)

sns.lineplot(
    df, x="ds", y="y", color="k", ax=ax[0]
)

ax[0].set_title("Walmart Sales in CA")
ax[0].set_ylabel("Sales")

sns.lineplot(
    (
        df
        .loc[
            (
                df.ds >= "2015-01-01"
            ) & (
                df.ds < "2016-01-01"
            )
        ]
    ),
    x="ds",
    y="y",
    color="k",
    ax=ax[1]
)

ax[1].set_title("Walmart Sales in CA in 2015")
ax[1].set_ylabel("Sales")

plot_acf(df.y, ax=ax[2], color="k");

# ----------------------------------------------------------------------

train_test_split_date = "2016-01-01"

df_train = df[df.ds<=train_test_split_date]
df_test = df[df.ds>train_test_split_date]

horizon = 3

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf = StatsForecast(
    models = [AutoARIMA(
        seasonal=False
    )],
    freq = 'd'
)

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
def forecast_test(
        forecaster_pred: callable,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> pd.DataFrame:
    """Generate forecasts for tests
    We gradually increase the historical information
    by appending part of the test data and 
    predict based on the new historical information.

    :param forecaster_pred: forecasting function/method
    :param df_train: train dataframe
    :param df_test: test dataframe
    """
    dfs_pred = []
    for i in range(len(df_test)):
        logger.debug(f"Prediction Step: {i}")
        df_pred_input_i = pd.concat(
            [df_train, df_test[:i]]
        ).reset_index(drop=True)
        df_pred_output_i = forecaster_pred(df_pred_input_i)
        df_pred_output_i["step"] = i
        dfs_pred.append(
            df_pred_output_i
        )
    df_y_hat = pd.concat(dfs_pred)

    return df_y_hat

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
df_y_hat_arima = forecast_test(
    lambda x: sf.forecast(df=x, h=horizon),
    df_train=df_train,
    df_test=df_test
)

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
from neuralforecast import NeuralForecast
from neuralforecast.models import VanillaTransformer, iTransformer

models = [
    VanillaTransformer(
        hidden_size=128,
        n_head=4,
        learning_rate=0.0001,
        scaler_type="robust",
        max_steps=500,
        batch_size=32,
        windows_batch_size=512,
        random_seed=16,
        input_size=30,
        step_size=3,
        h=horizon,
    ),
    iTransformer(
        input_size=30,
        h=horizon,
        max_steps=50,
        n_series=1,
    )
]
nf = NeuralForecast(models=models, freq="d")

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
nf.fit(df=df_train)

# ----------------------------------------------------------------------

#| echo: true
#| eval: false
df_y_hat_transformer = forecast_test(
    nf.predict, df_train=df_train, df_test=df_test
)