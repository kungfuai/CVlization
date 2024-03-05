import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as functional

from .lpips import LPIPS


def generator_loss(logits: torch.Tensor, loss_type: str = "hinge"):
    """
    :param logits: discriminator output in the generator phase (fake_logits)
    :param loss_type: which loss to apply between 'hinge' and 'non-saturating'
    """
    if loss_type == "hinge":
        loss = -torch.mean(logits)
    elif loss_type == "non-saturating":
        # Torch docs for bce with logits:
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability
        loss = functional.binary_cross_entropy_with_logits(
            logits, target=torch.ones_like(logits)
        )
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")
    return loss


def discriminator_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor, loss_type: str = "hinge"
):
    """
    :param logits_real: discriminator output when input is the original image
    :param logits_fake: discriminator output when input is the reconstructed image
    :param loss_type: which loss to apply between 'hinge' and 'non-saturating'
    """

    if loss_type == "hinge":
        real_loss = functional.relu(1.0 - logits_real)
        fake_loss = functional.relu(1.0 + logits_fake)
    elif loss_type == "non-saturating":
        # Torch docs for bce with logits:
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability
        real_loss = functional.binary_cross_entropy_with_logits(
            logits_real, target=torch.ones_like(logits_real), reduction="none"
        )
        fake_loss = functional.binary_cross_entropy_with_logits(
            logits_fake, target=torch.zeros_like(logits_fake), reduction="none"
        )
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    return torch.mean(real_loss + fake_loss)


class VQLPIPS(nn.Module):

    def __init__(self, l1_weight: float, l2_weight: float, perc_weight: float):
        """
        VQGAN Loss without discriminator. Used just for ablation.
        """

        super().__init__()

        self.l1_loss = lambda rec, tar: (tar - rec).abs().mean()
        self.l1_weight = l1_weight

        self.l2_loss = lambda rec, tar: (tar - rec).pow(2).mean()
        self.l2_weight = l2_weight

        self.perceptual_loss = LPIPS(net_type="alex")
        self.perceptual_weight = perc_weight

    def forward(
        self, quantizer_loss: float, images: torch.Tensor, reconstructions: torch.Tensor
    ):
        """
        :returns quant + nll loss, l1 loss, l2 loss, perceptual loss
        """

        # reconstruction losses
        l1_loss = self.l1_loss(reconstructions.contiguous(), images.contiguous())
        l2_loss = self.l2_loss(reconstructions.contiguous(), images.contiguous())
        p_loss = self.perceptual_loss(images.contiguous(), reconstructions.contiguous())

        nll_loss = (
            l1_loss * self.l1_weight
            + l2_loss * self.l2_weight
            + p_loss * self.perceptual_weight
        )

        loss = quantizer_loss + nll_loss

        return loss, l1_loss, l2_loss, p_loss
