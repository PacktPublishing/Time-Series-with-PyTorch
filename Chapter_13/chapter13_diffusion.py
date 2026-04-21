# Extracted from chapter13_diffusion.qmd
# Do not edit the source .qmd file directly.

import dataclasses
from functools import cached_property

import numpy as np
import torch

# ----------------------------------------------------------------------

@dataclasses.dataclass
class DiffusionProcessParams:
    """Parameter that defines a diffusion process.

    :param steps: Number of steps in the diffusion process.
    :param beta: Beta parameter for the diffusion process,
        representing the degree of noise added at each step. 
        A higher beta value corresponds to more noise being
        added, which accelerates the diffusion process.
    """
    steps: int
    beta: float

    @cached_property
    def alpha(self) -> float:
        r"""$\alpha = 1 - \beta$"""
        return 1.0 - self.beta

    @cached_property
    def beta_by_step(self) -> np.ndarray:
        """the beta parameter for each step
        in the diffusion process.
        """
        return np.array([self.beta] * self.steps)

    @cached_property
    def alpha_by_step(self) -> np.ndarray:
        """the alpha parameter for each step 
        in the diffusion process."""
        return np.array([self.alpha] * self.steps)

# ----------------------------------------------------------------------

class DiffusionProcess:
    """
    Diffusion process.

    :param params: DiffusionParams that defines 
        how the diffusion process works
    :param noise: noise tensor, 
        shape is (batch_size, params.steps)
    """

    def __init__(
        self,
        params: DiffusionProcessParams,
        noise: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.params = params
        self.noise = noise
        self.dtype = dtype

    @cached_property
    def alpha_by_step(self) -> torch.Tensor:
        """The alpha parameter for each step
        in the diffusion process.
        """
        return torch.tensor(
            self.params.alpha_by_step,
            dtype=self.dtype
        )

    def _forward_process_by_step(
        self, 
        state: torch.Tensor, 
        step: int
    ) -> torch.Tensor:
        r"""Assuming that we know 
        the noise at step $t$,

        $$
        x(t) = \sqrt{\alpha(t)}x(t-1) 
        + \sqrt{1 - \alpha(t)}\epsilon(t)
        $$

        :param state: The state at step $t-1$.
        :param step: The current step $t$.
        :return: The state at step $t$.
        """
        return (
            torch.sqrt(
                self.alpha_by_step[step]
            ) * state
            + torch.sqrt(
                1 - self.alpha_by_step[step]
            ) * self.noise[:, step]
        )

    def _inverse_process_by_step(
            self, 
            state: torch.Tensor, 
            step: int
        ) -> torch.Tensor:
        r"""Assuming that we know 
        the noise at step $t$,

        $$
        x(t-1) = \frac{1}{\sqrt{\alpha(t)}}
        (x(t) - \sqrt{1 - \alpha(t)}\epsilon(t))
        $$
        """
        return (
            state 
            - torch.sqrt(
                1 - self.alpha_by_step[step]
            ) * self.noise[:, step]
        ) / torch.sqrt(
            self.alpha_by_step[step]
        )

# ----------------------------------------------------------------------

def gaussian_noise(
        n_var: int, 
        length: int
    ) -> torch.Tensor:
    """Generate a Gaussian noise tensor.

    :param n_var: Number of variables.
    :param length: Length of the tensor.
    """
    return torch.normal(
        mean=0, 
        std=1, 
        size=(n_var, length)
    )

# ----------------------------------------------------------------------

diffusion_process_params = DiffusionProcessParams(
    steps=100,
    beta=0.005,
)
diffusion_batch_size = 1000

noise = gaussian_noise(
    diffusion_batch_size,
    diffusion_process_params.steps
)

diffusion_process = DiffusionProcess(
    diffusion_process_params,
    noise=noise
)

# ----------------------------------------------------------------------

#| message: false
#| echo: false
#| eval: true
diffusion_initial_x = torch.rand(diffusion_batch_size)

diffusion_steps_step_by_step = [
    diffusion_initial_x.detach().numpy()
]

for i in range(
    0, 
    diffusion_process_params.steps
):
    i_state = (
        diffusion_process._forward_process_by_step(
            torch.from_numpy(
                diffusion_steps_step_by_step[-1]
            ), 
            step=i
        )
        .detach()
        .numpy()
    )

    diffusion_steps_step_by_step.append(i_state)

# ----------------------------------------------------------------------

#| message: false
#| echo: false
#| eval: true
import pandas as pd


df_diffusion_example = pd.DataFrame(
    {i: v for i, v in enumerate(diffusion_steps_step_by_step)}
).T
df_diffusion_example["step"] = df_diffusion_example.index

df_diffusion_example_melted = df_diffusion_example.melt(
    id_vars=["step"], var_name="variable", value_name="value"
)

# ----------------------------------------------------------------------

#| label: fig-ts-diffusion-simulation-forward
#| fig-cap: "Histograms of particle positions at selected steps. On the right, the numbers indicate the corresponding step of the histogram. The distribution of the particles is more and more Gaussian as the diffusion process goes on. Meanwhile they are more dispersed, which also echos the name diffusion process."
#| echo: false
import matplotlib.pyplot as plt
import seaborn as sns

ridge_steps = 10


fig, ax = plt.subplots(figsize=(6, 6))
colors = plt.cm.viridis(np.linspace(
    0, 1, ridge_steps
))
bin_edges = np.histogram_bin_edges(
    df_diffusion_example_melted['value'], 
    bins='auto'
)

for i, step in enumerate(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
):
    values = df_diffusion_example_melted[
        df_diffusion_example_melted['step'] == step
    ]['value']
    counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts / counts.max()

    offset = i * 1.2
    ax.fill_between(
        bin_edges[:-1], offset, counts + offset, 
        step='mid', color=colors[i], alpha=0.7
    )
    ax.text(
        bin_edges[-1] + 0.2, offset, step, va='center')


ax.axes.get_yaxis().set_visible(False)

plt.axis('off')
ax.set_xlabel("Position")
plt.tight_layout()

# ----------------------------------------------------------------------

import torch.nn as nn


class DiffusionEncoder(nn.Module):
    """Encode the time series 
    in the latent space.

    :param params: parameters that
        define a diffusion process.
    :param noise: predefined noise 
        for the diffusion process.
    """
    def __init__(
        self,
        params: DiffusionProcessParams,
        noise: torch.Tensor,
    ):
        super().__init__()
        self.params = params
        self.noise = noise

    @staticmethod
    def _forward_process_by_step(
        state: torch.Tensor, 
        alpha_by_step: torch.Tensor, 
        noise: torch.Tensor, 
        step: int,
    ) -> torch.Tensor:
        r"""Assuming that we know the noise
        at step $t$,

        $$
        x(t) = \sqrt{\alpha(t)}x(t-1) 
        + \sqrt{1 - \alpha(t)}\epsilon(t)
        $$
        """
        batch_size = state.shape[0]
        return torch.sqrt(
            alpha_by_step[step]
        ) * state + (
            torch.sqrt(
                1 - alpha_by_step[step]
            ) * noise[:batch_size, step]
        ).reshape(batch_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Encoding the latent space 
        into a distribution.

        :param x: input data, shape
            (batch_size, history_length, n_features)
        """
        alpha_by_step = torch.tensor(
            self.params.alpha_by_step
        ).to(x)
        self.noise = self.noise.to(x)

        diffusion_steps_step_by_step = [x]

        for i in range(0, self.params.steps):
            i_state = self._forward_process_by_step(
                diffusion_steps_step_by_step[-1],
                alpha_by_step=alpha_by_step,
                noise=self.noise,
                step=i,
            )
            diffusion_steps_step_by_step.append(
                i_state
            )

        return diffusion_steps_step_by_step[-1]

# ----------------------------------------------------------------------

class DiffusionDecoder(nn.Module):
    """Decode the latent space into a distribution."""

    def __init__(
        self,
        params: DiffusionProcessParams,
        noise: torch.Tensor,
    ):
        super().__init__()
        self.params = params
        self.noise = noise

    @staticmethod
    def _inverse_process_by_step(
        state: torch.Tensor, 
        alpha_by_step: torch.Tensor, 
        noise: torch.Tensor, 
        step: int
    ) -> torch.Tensor:
        r"""Assuming that we know the 
        noise at step $t$,

        $$
        x(t-1) = \frac{1}{\sqrt{\alpha(t)}}
        (x(t) - \sqrt{1 - \alpha(t)}\epsilon(t))
        $$
        """
        batch_size = state.shape[0]
        return (
            state - (
                torch.sqrt(
                    1 - alpha_by_step[step]
                ) * noise[:batch_size, step]
            ).reshape(
                batch_size, 1
            )
        ) / torch.sqrt(alpha_by_step[step])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encoding the latent space into a distribution.

        :param x: input data, shape 
            (batch_size, history_length, n_features)
        """
        alpha_by_step = torch.tensor(
            self.params.alpha_by_step
        ).to(x)
        self.noise = self.noise.to(x)

        diffusion_steps_reverse = [x]

        for i in range(
            self.params.steps - 1, -1, -1
        ):
            i_state = self._inverse_process_by_step(
                state=diffusion_steps_reverse[-1],
                alpha_by_step=alpha_by_step,
                noise=self.noise,
                step=i,
            )
            diffusion_steps_reverse.append(i_state)

        return diffusion_steps_reverse[-1]

# ----------------------------------------------------------------------

@dataclasses.dataclass
class LatentRNNParams:
    """Parameters for Diffusion process.

    :param history_length: input sequence length
    :param latent_size: latent space dimension
    :param num_layers: number of RNN layers
    :param n_features: number of features in data
    :param initial_state: initial state of the RNN
    """
    history_length: int
    latent_size: int = 100
    num_layers: int = 2
    n_features: int = 1
    initial_state: torch.Tensor = None

    @cached_property
    def data_size(self) -> int:
        """The dimension of the input data
        when flattened.
        """
        return self.sequence_length * self.n_features

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

# ----------------------------------------------------------------------

class LatentRNN(nn.Module):
    """Forecasting the next step in latent space."""

    def __init__(self, params: LatentRNNParams):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()

        self.rnn = nn.GRU(
            input_size=self.params.history_length,
            hidden_size=self.params.latent_size,
            num_layers=self.params.num_layers,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data, shape 
            (batch_size, history_length * n_features)
        """
        outputs, _ = self.rnn(
            x, self.params.initial_state
        )

        return outputs

# ----------------------------------------------------------------------

class NaiveDiffusionModel(nn.Module):
    """A naive diffusion model that explicitly calculates
    the diffusion process.
    """
    def __init__(
        self,
        rnn: LatentRNN,
        diffusion_decoder: DiffusionDecoder,
        diffusion_encoder: DiffusionEncoder,
        horizon: int = 1,
    ):
        super().__init__()
        self.rnn = rnn
        self.diffusion_decoder = diffusion_decoder
        self.diffusion_encoder = diffusion_encoder
        self.horizon = horizon
        self.scale = nn.Linear(
            in_features=self.rnn.params.latent_size,
            out_features=self.horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_latent = self.diffusion_encoder(x.squeeze(-1))
        y_latent = self.rnn(x_latent)
        y_hat = self.diffusion_decoder(y_latent)

        y_hat = self.scale(y_hat)

        return y_hat

# ----------------------------------------------------------------------

import lightning.pytorch as pl
from lightning import LightningModule

class NaiveDiffusionForecaster(LightningModule):
    """A assembled lightning module 
    for the naive diffusion model.
    """
    def __init__(
        self,
        model: NaiveDiffusionModel,
        loss: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.model = model
        self.loss = loss

    def configure_optimizers(
        self
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=1e-3
        )
        return optimizer

    def training_step(
        self, 
        batch: tuple[torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(
            x
        )[:batch_size, :].reshape_as(y)

        loss = self.loss(y_hat, y).mean()
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
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(
            x
        )[:batch_size, :].reshape_as(y)

        loss = self.loss(y_hat, y).mean()
        self.log_dict(
            {"val_loss": loss},
            prog_bar=True
        )
        return loss

    def predict_step(
        self, 
        batch: tuple[torch.Tensor],
        batch_idx: int
    ) -> tuple[
        torch.Tensor, 
        torch.Tensor
    ]:
        x, y = batch
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(
            x
        )[:batch_size, :].reshape_as(y)
        return x, y_hat

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(self.model.rnn.rnn.weight_ih_l0)
        return self.model(x)

# ----------------------------------------------------------------------

history_length_1_step = 100
horizon_1_step = 1
training_batch_size = 64

training_noise = gaussian_noise(
    training_batch_size,
    diffusion_process_params.steps
)

# ----------------------------------------------------------------------

diffusion_decoder = DiffusionDecoder(
    diffusion_process_params,
    training_noise
)
diffusion_encoder = DiffusionEncoder(
    diffusion_process_params,
    training_noise
)

latent_rnn_params = LatentRNNParams(
    history_length=history_length_1_step,
    latent_size=diffusion_process_params.steps,
)

latent_rnn = LatentRNN(latent_rnn_params)

# ----------------------------------------------------------------------

naive_diffusion_model = NaiveDiffusionModel(
    rnn=latent_rnn,
    diffusion_decoder=diffusion_decoder,
    diffusion_encoder=diffusion_encoder,
)
naive_diffusion_forecaster = NaiveDiffusionForecaster(
    model=naive_diffusion_model.float(),
)