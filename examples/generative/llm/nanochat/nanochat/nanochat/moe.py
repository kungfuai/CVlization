"""
Mixture of Experts (MoE) implementation for nanochat.

Design choices (following DeepSeek V3 / Kimi K2 patterns):
- Sigmoid routing with top-k selection
- Optional shared expert (always-on)
- Aux-loss-free load balancing via bias term
- Fine-grained experts (many small experts)
- Batched expert computation for efficiency

Usage:
    # In GPTConfig, set use_moe=True
    config = GPTConfig(use_moe=True, num_experts=8, num_experts_per_tok=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Single expert MLP - same architecture as the standard MLP."""

    def __init__(self, n_embd: int, expert_dim: int = None):
        super().__init__()
        expert_dim = expert_dim or 4 * n_embd
        self.c_fc = nn.Linear(n_embd, expert_dim, bias=False)
        self.c_proj = nn.Linear(expert_dim, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2 activation (same as standard MLP)
        x = self.c_proj(x)
        return x


class FusedExperts(nn.Module):
    """
    Memory-efficient fused expert computation.

    Stores all expert weights in single tensors but processes experts sequentially
    to avoid memory explosion during gather operations. Each expert processes its
    batch of tokens efficiently with standard matmul.
    """

    def __init__(self, n_embd: int, expert_dim: int, num_experts: int):
        super().__init__()
        self.n_embd = n_embd
        self.expert_dim = expert_dim
        self.num_experts = num_experts

        # Fused weights: (num_experts, expert_dim, n_embd) for up-projection
        # and (num_experts, n_embd, expert_dim) for down-projection
        self.w1 = nn.Parameter(torch.empty(num_experts, expert_dim, n_embd))
        self.w2 = nn.Parameter(torch.empty(num_experts, n_embd, expert_dim))

    def forward(self, x, expert_indices, expert_weights):
        """
        Memory-efficient batched expert forward pass.

        Processes experts sequentially but uses efficient batched matmul within
        each expert. This avoids the memory explosion of expanding weights to
        (B*T, k, expert_dim, n_embd).

        Args:
            x: Input tensor (B*T, D)
            expert_indices: Selected expert indices (B*T, k)
            expert_weights: Weights for selected experts (B*T, k)

        Returns:
            Output tensor (B*T, D)
        """
        num_tokens = x.shape[0]
        k = expert_indices.shape[1]
        device = x.device
        dtype = expert_weights.dtype

        # Output accumulator
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find (token, slot) pairs where this expert is selected
            # mask: (B*T, k) - True where expert_indices == expert_idx
            mask = (expert_indices == expert_idx)

            if not mask.any():
                continue

            # Get unique tokens that use this expert (may appear in multiple slots)
            token_mask = mask.any(dim=-1)  # (B*T,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]  # indices of tokens using this expert

            if len(token_indices) == 0:
                continue

            # Get input for this expert
            expert_input = x[token_indices]  # (n_tokens, D)

            # Up projection: expert_input @ w1[expert_idx].T
            # expert_input: (n_tokens, D), w1[expert_idx]: (expert_dim, D)
            hidden = expert_input @ self.w1[expert_idx].T  # (n_tokens, expert_dim)

            # Activation: relu^2
            hidden = F.relu(hidden).square()

            # Down projection: hidden @ w2[expert_idx].T
            # hidden: (n_tokens, expert_dim), w2[expert_idx]: (D, expert_dim)
            expert_output = hidden @ self.w2[expert_idx].T  # (n_tokens, D)

            # Get weights for this expert (summed across slots if token uses this expert multiple times)
            weights = torch.zeros(len(token_indices), device=device, dtype=dtype)
            for slot_idx in range(k):
                slot_mask = mask[token_indices, slot_idx]
                weights[slot_mask] += expert_weights[token_indices[slot_mask], slot_idx]

            # Accumulate weighted output
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        return output


class Router(nn.Module):
    """
    Top-k router with sigmoid gating (DeepSeek V3 style).

    Uses sigmoid affinities for selection, then normalizes weights
    over selected experts only.
    """

    def __init__(self, n_embd: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router projection: input -> expert scores
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

        # Learnable bias for load balancing (aux-loss-free approach)
        # This affects routing selection but NOT the gating weights
        self.load_balance_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            expert_weights: (B, T, k) - normalized weights for selected experts
            expert_indices: (B, T, k) - indices of selected experts
        """
        # Compute router logits
        logits = self.gate(x)  # (B, T, num_experts)

        # Add load balance bias for selection (but not for weights)
        selection_logits = logits + self.load_balance_bias

        # Select top-k experts based on biased logits
        _, expert_indices = torch.topk(selection_logits, self.num_experts_per_tok, dim=-1)

        # Compute weights using sigmoid on ORIGINAL logits (not biased)
        # Gather the logits for selected experts
        selected_logits = torch.gather(logits, dim=-1, index=expert_indices)

        # Sigmoid + normalize over selected experts
        expert_weights = torch.sigmoid(selected_logits)
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-6)

        return expert_weights, expert_indices


class MoELayer(nn.Module):
    """
    Mixture of Experts layer that replaces the standard MLP.

    Features:
    - Multiple expert MLPs with top-k routing
    - Optional shared expert (always active)
    - Batched expert computation for efficiency
    - Optional auxiliary load balance loss
    """

    def __init__(self, n_embd: int, num_experts: int = 8, num_experts_per_tok: int = 2,
                 use_shared_expert: bool = True, expert_dim: int = None,
                 aux_loss_weight: float = 0.01):
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = use_shared_expert
        self.aux_loss_weight = aux_loss_weight

        # Expert dimension - smaller per expert since we have many
        # With more experts, we use smaller expert_dim for better specialization
        if expert_dim is None:
            # Scale down expert dim so active compute is ~half of standard MLP
            # Standard MLP: 4 * n_embd intermediate, so 8 * n_embd² params
            # MoE: k * expert_dim * n_embd * 2 params active per token
            # For k=2, expert_dim = n_embd gives 4 * n_embd² active (half of dense)
            # This allows more experts (16+) while keeping compute reasonable
            expert_dim = (2 * n_embd) // num_experts_per_tok
        self.expert_dim = expert_dim

        # Router
        self.router = Router(n_embd, num_experts, num_experts_per_tok)

        # Fused experts for batched computation
        self.fused_experts = FusedExperts(n_embd, expert_dim, num_experts)

        # Shared expert (optional, always-on)
        if use_shared_expert:
            self.shared_expert = ExpertMLP(n_embd, expert_dim)
        else:
            self.shared_expert = None

        # Auxiliary loss accumulator (set during forward, read during training)
        self._aux_loss = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Get routing weights and indices
        expert_weights, expert_indices = self.router(x)  # (B, T, k), (B, T, k)

        # Flatten for batched computation
        flat_x = x.view(-1, D)  # (B*T, D)
        flat_weights = expert_weights.view(-1, self.num_experts_per_tok)  # (B*T, k)
        flat_indices = expert_indices.view(-1, self.num_experts_per_tok)  # (B*T, k)

        # Batched expert computation
        output = self.fused_experts(flat_x, flat_indices, flat_weights)
        output = output.view(B, T, D)

        # Add shared expert contribution (if enabled)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            output = output + shared_output

        # Compute auxiliary load balance loss (if enabled)
        if self.aux_loss_weight > 0 and self.training:
            self._aux_loss = self._compute_aux_loss(expert_indices, B * T)
        else:
            self._aux_loss = None

        return output

    def _compute_aux_loss(self, expert_indices, num_tokens):
        """
        Compute auxiliary load balance loss to encourage even expert usage.

        Following OLMoE: loss = num_experts * sum(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
        and P_i = mean router probability for expert i
        """
        # Count how often each expert is selected
        # expert_indices: (B, T, k) -> flatten to (B*T*k,)
        flat_indices = expert_indices.view(-1)
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()

        # Fraction of (token, slot) pairs routed to each expert
        total_assignments = flat_indices.numel()
        f = expert_counts / total_assignments  # (num_experts,)

        # For auxiliary loss, we want to balance f_i across experts
        # Ideal: each expert gets 1/num_experts of assignments
        # Loss: sum of squared deviations from uniform
        # Simpler formulation: num_experts * sum(f_i^2) - 1
        # This is minimized when f is uniform
        aux_loss = self.num_experts * (f.pow(2).sum()) - 1.0

        return self.aux_loss_weight * aux_loss

    def get_aux_loss(self):
        """Get the auxiliary loss from the last forward pass."""
        return self._aux_loss if self._aux_loss is not None else 0.0


def init_moe_weights(moe_layer: MoELayer):
    """Initialize MoE layer weights following nanochat conventions."""
    n_embd = moe_layer.n_embd
    s = 3**0.5 * n_embd**-0.5

    # Router
    torch.nn.init.uniform_(moe_layer.router.gate.weight, -s, s)
    torch.nn.init.zeros_(moe_layer.router.load_balance_bias)

    # Fused experts: w1 (up-projection), w2 (down-projection)
    # w1: (num_experts, expert_dim, n_embd) - uniform init
    # w2: (num_experts, n_embd, expert_dim) - zero init (residual stream)
    torch.nn.init.uniform_(moe_layer.fused_experts.w1, -s, s)
    torch.nn.init.zeros_(moe_layer.fused_experts.w2)

    # Shared expert
    if moe_layer.shared_expert is not None:
        torch.nn.init.uniform_(moe_layer.shared_expert.c_fc.weight, -s, s)
        torch.nn.init.zeros_(moe_layer.shared_expert.c_proj.weight)
