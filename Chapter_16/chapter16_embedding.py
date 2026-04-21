# Extracted from chapter16_embedding.qmd
# Do not edit the source .qmd file directly.

from torch.utils.data import DataLoader
import lightning as L
import torch

def embedding_extractor(
    forecaster: L.LightningModule, 
    x: torch.Tensor
) -> tuple[torch.Tensor]:
    """compute the embeddings based on the input

    :param forecaster: the trained forecaster
    :param x: input historical time series,
    """
    forecaster.model.to(x.device)
    z_mean, z_log_var, z = forecaster.model.encoder(
        x.type_as(
            forecaster.model.encoder.z_mean_layer.weight
        )
    )
    
    return z_mean, z_log_var, z

# ----------------------------------------------------------------------

# | eval: false
z_mean_example, _, _ = embedding_extractor(
    vae_model,
    input_example
)

# ----------------------------------------------------------------------

# | eval: false
from torchdr import TSNE

n_components = 2

dr_input_result = TSNE(
    perplexity=30,
    n_components=n_components
).fit_transform(
    input_example.squeeze()
)

dr_z_result = TSNE(
    perplexity=30,
    n_components=n_components
).fit_transform(
    z_mean_example.squeeze()
)