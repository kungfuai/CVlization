import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch import nn
from torch.nn import functional as F


class EnergyBasedModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 0.0001):
        super().__init__()
        self.net = model
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def synthezise(self, num_samples: int=100) -> torch.Tensor:
        """
        Sample from the model.
        """
        raise NotImplementedError()
    
    def analyse(self, x, x_synth):
        """
        Compute loss for updating model parameters.
        """
        avg_energy_real_data = self(x).mean()
        avg_energy_sampled_data = self(x_synth).mean()
        loss = avg_energy_sampled_data - avg_energy_real_data
        return loss

    def training_step(self, batch, batch_idx=None):
        x = batch
        x_synth = self.synthezise()
        loss = self.analyse(x, x_synth)
        # Note: The return value can be None, a loss tensor, or a dictionary with a "loss" key.
        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        return {"loss": val_loss}