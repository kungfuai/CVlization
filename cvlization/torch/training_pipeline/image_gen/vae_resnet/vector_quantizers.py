from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, einsum


class BaseVectorQuantizer(ABC, nn.Module):
    """
    What does it do? It quantizes the input tensor to a set of discrete values (codebook).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        """

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # wu/decay init (may never be used)
        self.kl_warmup = None
        self.temp_decay = None

    def init_codebook(self) -> None:
        """
        uniform initialization of the codebook
        """
        nn.init.uniform_(
            self.codebook.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.IntTensor, float]:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """
        pass

    @abstractmethod
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        pass

    @torch.no_grad()
    def get_codebook(self) -> torch.nn.Embedding:
        return self.codebook.weight

    @torch.no_grad()
    def codes_to_vec(self, codes: torch.IntTensor) -> torch.Tensor:
        """
        :param codes: int tensors to decode (B, N).
        :return flat codebook indices (B, N, D)
        """

        quantized = self.get_codebook()[codes]
        return quantized

    def get_codebook_usage(self, index_count: torch.Tensor):
        """
        :param index_count: (n, ) where n is the codebook size, express the number of times each index have been used.
        :return: prob of each index to be used: (n, ); perplexity: float; codebook_usage: float 0__1
        """

        # get used idx as probabilities
        used_indices = index_count / torch.sum(index_count)

        # perplexity
        perplexity = (
            torch.exp(
                -torch.sum(used_indices * torch.log(used_indices + 1e-10), dim=-1)
            )
            .sum()
            .item()
        )

        # get the percentage of used codebook
        n = index_count.shape[0]
        used_codebook = (torch.count_nonzero(used_indices).item() * 100) / n

        return used_indices, perplexity, used_codebook

    @torch.no_grad()
    def reinit_unused_codes(self, codebook_usage: torch.Tensor):
        """
        Re-initialize unused vectors according to the likelihood of used ones.
        :param codebook_usage: (n, ) where n is the codebook size, distribution probability of codebook usage.
        """

        device = codebook_usage.device
        n = codebook_usage.shape[0]

        # compute unused codes
        unused_codes = torch.nonzero(
            torch.eq(codebook_usage, torch.zeros(n, device=device))
        ).squeeze(1)
        n_unused = unused_codes.shape[0]

        # sample according to most used codes.
        # torch.use_deterministic_algorithms(False)
        replacements = torch.multinomial(codebook_usage, n_unused, replacement=True)
        # torch.use_deterministic_algorithms(True)

        # update unused codes
        new_codes = self.codebook.weight[replacements]
        self.codebook.weight[unused_codes] = new_codes


class VectorQuantizer(BaseVectorQuantizer):

    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
    ):
        """
        Original VectorQuantizer with straight through gradient estimator (loss is optimized on inputs and codebook)
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param commitment_cost: scaling factor for e_loss
        """
        print(
            f"***** Creating a quantizer with {num_embeddings} embeddings and {embedding_dim} dimensions"
        )
        super().__init__(num_embeddings, embedding_dim)

        self.commitment_cost = commitment_cost

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.IntTensor, float]:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """
        b, c, h, w = x.shape
        device = x.device

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c h w -> (b h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Loss functions
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)
        q_loss = F.mse_loss(quantized, flat_x.detach())

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        quantized = rearrange(quantized, "(b h w) c -> b c h w", b=b, h=h, w=w)
        encoding_indices = rearrange(
            encoding_indices, "(b h w)-> b (h w)", b=b, h=h, w=w
        ).detach()

        return quantized, encoding_indices, q_loss + e_loss

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        b, c, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c h w -> (b h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(
            encoding_indices, "(b h w) -> b (h w)", b=b, h=h, w=w
        )

        return encoding_indices


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        straight_through: bool = False,
        temp: float = 1.0,
        kl_cost: float = 5e-4,
    ):
        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param straight_through: if True, will one-hot quantize, but still differentiate as if it is the soft sample
        :param temp: temperature parameter for gumbel softmax
        :param kl_cost: cost for kl divergence
        """
        super().__init__(num_embeddings, embedding_dim)

        self.x_to_logits = torch.nn.Conv2d(num_embeddings, num_embeddings, 1)
        self.straight_through = straight_through
        self.temp = temp
        self.kl_cost = kl_cost

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,N,H,W). Note that N = number of embeddings in dict!
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        # deterministic quantization during inference
        hard = self.straight_through if self.training else True

        logits = self.x_to_logits(x)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temp, dim=1, hard=hard)
        quantized = einsum(soft_one_hot, self.get_codebook(), "b n h w, n d -> b d h w")

        # + kl divergence to the prior (uniform) loss, increase cb usage
        # Note:
        #       KL(P(x), Q(x)) = sum_x (P(x) * log(P(x) / Q(x)))
        #       in this case: P(x) is qy, Q(x) is uniform distribution (1 / num_embeddings)
        qy = F.softmax(logits, dim=1)
        kl_loss = (
            self.kl_cost
            * torch.sum(qy * torch.log(qy * self.num_embeddings + 1e-10), dim=1).mean()
        )

        encoding_indices = soft_one_hot.argmax(dim=1).detach()

        return quantized, encoding_indices, kl_loss

    def get_consts(self) -> (float, float):
        """
        return temp, kl_cost
        """
        return self.temp, self.kl_cost

    def set_consts(self, temp: float = None, kl_cost: float = None) -> None:
        """
        update values for temp, kl_cost
        :param temp: new value for temperature (if not None)
        :param kl_cost: new value for kl_cost (if not None)
        """
        if temp is not None:
            self.temp = temp

        if kl_cost is not None:
            self.kl_cost = kl_cost

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,N,H,W). Note that N = number of embeddings in dict!
        :return flat codebook indices (B, H * W)
        """

        soft_one_hot = F.gumbel_softmax(x, tau=1.0, dim=1, hard=True)
        encoding_indices = soft_one_hot.argmax(dim=1)
        return encoding_indices
