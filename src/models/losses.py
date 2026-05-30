"""
This module implements:

1. Conditional flow-matching generation loss.
2. Progression smoothness regularization.
3. Ordinal severity regularization.
4. A combined L-Flow loss wrapper.

The full objective follows:

    L = L_gen + lambda_prog * L_prog + lambda_ord * L_ord

where:
    L_gen  : flow-matching vector field prediction loss
    L_prog : temporal/progression smoothness regularization
    L_ord  : ordinal severity structure regularization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class LFlowLossConfig:
    """
    Configuration for the full L-Flow loss.

    Attributes
    ----------
    lambda_gen:
        Weight for the flow-matching generation loss.
    lambda_prog:
        Weight for progression smoothness regularization.
    lambda_ord:
        Weight for ordinal severity regularization.
    generation_loss:
        Type of generation loss. Options: "mse", "l1", "smooth_l1".
    progression_distance:
        Distance used for progression regularization. Options: "l2", "l1", "cosine".
    ordinal_margin:
        Margin used in ordinal regularization.
    ordinal_mode:
        Type of ordinal regularization. Options: "pairwise_margin", "rank_correlation".
    use_masked_region_only:
        If True, compute generation loss only on the masked region.
    """

    lambda_gen: float = 1.0
    lambda_prog: float = 0.1
    lambda_ord: float = 0.1

    generation_loss: str = "mse"
    progression_distance: str = "l2"

    ordinal_margin: float = 0.1
    ordinal_mode: str = "pairwise_margin"

    use_masked_region_only: bool = True


# ---------------------------------------------------------------------
# Basic utility functions
# ---------------------------------------------------------------------


def masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute mean with an optional mask.

    Parameters
    ----------
    values:
        Tensor of arbitrary shape.
    mask:
        Tensor broadcastable to values.
        1 = include, 0 = ignore.
    eps:
        Numerical stability term.

    Returns
    -------
    torch.Tensor
        Scalar mean.
    """

    if mask is None:
        return values.mean()

    mask = mask.to(device=values.device, dtype=values.dtype)

    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)

    values = values * mask
    denom = mask.sum().clamp_min(eps)

    return values.sum() / denom


def pairwise_differences(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise differences.

    Input:
        x: [B]

    Output:
        diff: [B, B], where diff[i, j] = x[i] - x[j]
    """

    return x.unsqueeze(1) - x.unsqueeze(0)


def pairwise_distances(
    z: torch.Tensor,
    distance: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute pairwise distances between latent vectors.

    Parameters
    ----------
    z:
        Tensor of shape [B, D].
    distance:
        "l2", "l1", or "cosine".

    Returns
    -------
    Tensor of shape [B, B].
    """

    if z.ndim != 2:
        raise ValueError(f"Expected z shape [B, D], got {tuple(z.shape)}.")

    if distance == "l2":
        return torch.cdist(z, z, p=2)

    if distance == "l1":
        return torch.cdist(z, z, p=1)

    if distance == "cosine":
        z_norm = F.normalize(z, dim=-1, eps=eps)
        similarity = torch.matmul(z_norm, z_norm.transpose(0, 1))
        return 1.0 - similarity

    raise ValueError(
        f"Unknown distance: {distance}. Expected 'l2', 'l1', or 'cosine'."
    )


# ---------------------------------------------------------------------
# Generation loss
# ---------------------------------------------------------------------


class FlowMatchingLoss(nn.Module):
    """
    Conditional flow-matching generation loss.

    The model predicts the vector field along the linear path between Gaussian
    noise and the target mel-spectrogram. The target vector field is:

        v_target = target_mel - epsilon

    This loss compares the predicted vector field with the target vector field.

    Expected shapes:
        predicted_vector_field: [B, F, N]
        target_vector_field:    [B, F, N]
        mask:                   [B, F, N] or None

    In this codebase, mask value 1 means the region is masked/reconstructed.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        use_masked_region_only: bool = True,
    ) -> None:
        super().__init__()

        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(
                f"Unknown generation loss type: {loss_type}. "
                "Expected 'mse', 'l1', or 'smooth_l1'."
            )

        self.loss_type = loss_type
        self.use_masked_region_only = use_masked_region_only

    def forward(
        self,
        predicted_vector_field: torch.Tensor,
        target_vector_field: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if predicted_vector_field.shape != target_vector_field.shape:
            raise ValueError(
                "predicted_vector_field and target_vector_field must have the same shape. "
                f"Got {tuple(predicted_vector_field.shape)} and "
                f"{tuple(target_vector_field.shape)}."
            )

        if self.loss_type == "mse":
            loss = (predicted_vector_field - target_vector_field) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(predicted_vector_field - target_vector_field)
        else:
            loss = F.smooth_l1_loss(
                predicted_vector_field,
                target_vector_field,
                reduction="none",
            )

        if self.use_masked_region_only and mask is not None:
            return masked_mean(loss, mask)

        return loss.mean()


# ---------------------------------------------------------------------
# Progression smoothness regularization
# ---------------------------------------------------------------------


class ProgressionSmoothnessLoss(nn.Module):
    """
    Progression smoothness regularization.

    This loss encourages progression representations that are close in time
    to also be close in latent space.

    Given:
        z_prog:     [B, D]
        query_time: [B]

    We compute pairwise latent distances and weight them by temporal proximity:

        weight_ij = exp(-|t_i - t_j| / temperature)

    Then:

        L_prog = mean_ij weight_ij * distance(z_i, z_j)

    This is a practical implementation of the idea that temporally nearby
    progression states should remain close.
    """

    def __init__(
        self,
        distance: str = "l2",
        temperature: float = 1.0,
        exclude_self_pairs: bool = True,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        if distance not in {"l2", "l1", "cosine"}:
            raise ValueError(
                f"Unknown distance: {distance}. Expected 'l2', 'l1', or 'cosine'."
            )

        self.distance = distance
        self.temperature = temperature
        self.exclude_self_pairs = exclude_self_pairs

    def forward(
        self,
        z_prog: torch.Tensor,
        query_time: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_prog:
            Tensor of shape [B, D].
        query_time:
            Tensor of shape [B].
        valid_mask:
            Optional tensor of shape [B], where 1 means valid sample.

        Returns
        -------
        torch.Tensor
            Scalar progression smoothness loss.
        """

        if z_prog.ndim != 2:
            raise ValueError(f"Expected z_prog shape [B, D], got {tuple(z_prog.shape)}.")

        batch_size = z_prog.shape[0]

        if batch_size <= 1:
            return z_prog.new_tensor(0.0)

        query_time = query_time.view(batch_size).to(
            device=z_prog.device,
            dtype=z_prog.dtype,
        )

        latent_dist = pairwise_distances(z_prog, distance=self.distance)
        time_dist = torch.abs(pairwise_differences(query_time))

        # Normalize time distances per batch for scale stability.
        time_scale = time_dist.detach().max().clamp_min(1.0)
        time_dist = time_dist / time_scale

        weights = torch.exp(-time_dist / self.temperature)

        pair_mask = torch.ones_like(weights)

        if self.exclude_self_pairs:
            pair_mask = pair_mask * (1.0 - torch.eye(batch_size, device=z_prog.device))

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=z_prog.device, dtype=z_prog.dtype).view(batch_size)
            pair_mask = pair_mask * valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)

        weighted_loss = latent_dist * weights * pair_mask

        denom = pair_mask.sum().clamp_min(1.0)
        return weighted_loss.sum() / denom


# ---------------------------------------------------------------------
# Ordinal severity regularization
# ---------------------------------------------------------------------


class OrdinalSeverityLoss(nn.Module):
    """
    Ordinal severity regularization.

    This loss encourages the latent progression space to preserve the ordering
    implied by clinical severity labels.

    Main mode: pairwise_margin

    For each pair (i, j), if y_i > y_j, then the projection score of z_i should
    be larger than the projection score of z_j by at least a margin:

        score_i - score_j >= margin * |y_i - y_j|

    The loss is:

        max(0, margin * |y_i - y_j| - sign(y_i - y_j) * (score_i - score_j))

    This gives the model a differentiable ranking constraint.
    """

    def __init__(
        self,
        hidden_dim: int,
        margin: float = 0.1,
        mode: str = "pairwise_margin",
        min_label_gap: float = 1e-6,
    ) -> None:
        super().__init__()

        if mode not in {"pairwise_margin", "rank_correlation"}:
            raise ValueError(
                f"Unknown ordinal mode: {mode}. "
                "Expected 'pairwise_margin' or 'rank_correlation'."
            )

        self.hidden_dim = hidden_dim
        self.margin = margin
        self.mode = mode
        self.min_label_gap = min_label_gap

        # Learnable projection from latent state to scalar progression score.
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_prog: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_prog.ndim != 2:
            raise ValueError(f"Expected z_prog shape [B, D], got {tuple(z_prog.shape)}.")

        batch_size, hidden_dim = z_prog.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )

        if batch_size <= 1:
            return z_prog.new_tensor(0.0)

        labels = labels.view(batch_size).to(
            device=z_prog.device,
            dtype=z_prog.dtype,
        )

        scores = self.scorer(z_prog).squeeze(-1)

        if self.mode == "pairwise_margin":
            return self._pairwise_margin_loss(scores, labels, valid_mask)

        return self._rank_correlation_loss(scores, labels, valid_mask)

    def _pairwise_margin_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = scores.shape[0]

        label_diff = pairwise_differences(labels)
        score_diff = pairwise_differences(scores)

        label_gap = label_diff.abs()
        pair_sign = torch.sign(label_diff)

        # Use only pairs with a meaningful label difference.
        pair_mask = (label_gap > self.min_label_gap).to(scores.dtype)

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=scores.device, dtype=scores.dtype).view(batch_size)
            pair_mask = pair_mask * valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)

        # Remove self pairs.
        pair_mask = pair_mask * (1.0 - torch.eye(batch_size, device=scores.device))

        required_margin = self.margin * label_gap
        ranking_agreement = pair_sign * score_diff

        loss = F.relu(required_margin - ranking_agreement)
        loss = loss * pair_mask

        denom = pair_mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _rank_correlation_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable correlation-style ranking loss.

        This does not directly optimize Spearman's rho, but encourages the
        centered score vector to align with the centered label vector.

        Loss:
            1 - cosine(centered_scores, centered_labels)
        """

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=scores.device, dtype=scores.dtype)
            scores = scores * valid_mask
            labels = labels * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)

            scores = scores - scores.sum() / denom
            labels = labels - labels.sum() / denom

            scores = scores * valid_mask
            labels = labels * valid_mask
        else:
            scores = scores - scores.mean()
            labels = labels - labels.mean()

        score_norm = F.normalize(scores, dim=0)
        label_norm = F.normalize(labels, dim=0)

        corr = torch.sum(score_norm * label_norm)
        return 1.0 - corr


# ---------------------------------------------------------------------
# Combined L-Flow loss
# ---------------------------------------------------------------------


class LFlowLoss(nn.Module):
    """
    Combined L-Flow training objective.

    Expected model outputs:
        predicted_vector_field:
            [B, F, N]
        z_prog:
            [B, D]

    Expected batch fields:
        target_vector_field:
            [B, F, N]
        mask:
            [B, F, N]
        target_time:
            [B]
        target_label:
            [B]

    Returns a dictionary:
        {
            "loss": total_loss,
            "loss_gen": generation_loss,
            "loss_prog": progression_loss,
            "loss_ord": ordinal_loss
        }
    """

    def __init__(
        self,
        hidden_dim: int,
        config: Optional[LFlowLossConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or LFlowLossConfig()

        self.generation_loss = FlowMatchingLoss(
            loss_type=self.config.generation_loss,
            use_masked_region_only=self.config.use_masked_region_only,
        )

        self.progression_loss = ProgressionSmoothnessLoss(
            distance=self.config.progression_distance,
        )

        self.ordinal_loss = OrdinalSeverityLoss(
            hidden_dim=hidden_dim,
            margin=self.config.ordinal_margin,
            mode=self.config.ordinal_mode,
        )

    def forward(
        self,
        predicted_vector_field: torch.Tensor,
        target_vector_field: torch.Tensor,
        z_prog: torch.Tensor,
        target_time: torch.Tensor,
        target_label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_gen = self.generation_loss(
            predicted_vector_field=predicted_vector_field,
            target_vector_field=target_vector_field,
            mask=mask,
        )

        if self.config.lambda_prog > 0:
            loss_prog = self.progression_loss(
                z_prog=z_prog,
                query_time=target_time,
                valid_mask=valid_mask,
            )
        else:
            loss_prog = z_prog.new_tensor(0.0)

        if self.config.lambda_ord > 0:
            loss_ord = self.ordinal_loss(
                z_prog=z_prog,
                labels=target_label,
                valid_mask=valid_mask,
            )
        else:
            loss_ord = z_prog.new_tensor(0.0)

        total_loss = (
            self.config.lambda_gen * loss_gen
            + self.config.lambda_prog * loss_prog
            + self.config.lambda_ord * loss_ord
        )

        return {
            "loss": total_loss,
            "loss_gen": loss_gen.detach(),
            "loss_prog": loss_prog.detach(),
            "loss_ord": loss_ord.detach(),
        }


# ---------------------------------------------------------------------
# Auxiliary loss functions for evaluation/debugging
# ---------------------------------------------------------------------


def compute_pairwise_order_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_label_gap: float = 1e-6,
) -> torch.Tensor:
    """
    Compute pairwise ordering accuracy.

    This can be used during validation to check whether predicted scalar scores
    preserve severity ordering.

    Parameters
    ----------
    scores:
        Predicted scalar progression scores, shape [B].
    labels:
        Ground-truth labels, shape [B].

    Returns
    -------
    torch.Tensor
        Scalar pairwise accuracy.
    """

    scores = scores.view(-1)
    labels = labels.view(-1)

    if scores.numel() <= 1:
        return scores.new_tensor(0.0)

    label_diff = pairwise_differences(labels)
    score_diff = pairwise_differences(scores)

    valid_pairs = label_diff.abs() > min_label_gap

    # Remove duplicate/self direction by using upper triangular mask.
    upper = torch.triu(
        torch.ones_like(valid_pairs, dtype=torch.bool),
        diagonal=1,
    )
    valid_pairs = valid_pairs & upper

    if valid_pairs.sum() == 0:
        return scores.new_tensor(0.0)

    correct = torch.sign(label_diff[valid_pairs]) == torch.sign(score_diff[valid_pairs])
    return correct.float().mean()


def compute_latent_temporal_smoothness(
    z_prog: torch.Tensor,
    query_time: torch.Tensor,
    distance: str = "l2",
) -> torch.Tensor:
    """
    Diagnostic metric: average latent distance between temporally adjacent items.

    This assumes the batch can be sorted by query_time.
    """

    if z_prog.shape[0] <= 1:
        return z_prog.new_tensor(0.0)

    order = torch.argsort(query_time.view(-1))
    z_sorted = z_prog[order]

    diffs = z_sorted[1:] - z_sorted[:-1]

    if distance == "l2":
        return torch.norm(diffs, p=2, dim=-1).mean()

    if distance == "l1":
        return torch.norm(diffs, p=1, dim=-1).mean()

    if distance == "cosine":
        z1 = F.normalize(z_sorted[1:], dim=-1)
        z0 = F.normalize(z_sorted[:-1], dim=-1)
        return (1.0 - (z1 * z0).sum(dim=-1)).mean()

    raise ValueError(f"Unknown distance: {distance}")


# ---------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------


def build_loss_from_config(config: dict) -> LFlowLoss:
    """
    Build LFlowLoss from loaded YAML config.

    Expected config sections:

    model:
      progression_encoder:
        hidden_dim: 256

    loss:
      lambda_gen: 1.0
      lambda_prog: 0.1
      lambda_ord: 0.1
      progression_regularization:
        distance: "l2"
      ordinal_regularization:
        margin: 0.1
    """

    model_cfg = config.get("model", {})
    enc_cfg = model_cfg.get("progression_encoder", {})
    hidden_dim = int(enc_cfg.get("hidden_dim", 256))

    loss_cfg = config.get("loss", {})

    prog_cfg = loss_cfg.get("progression_regularization", {})
    ord_cfg = loss_cfg.get("ordinal_regularization", {})

    flow_cfg = config.get("flow_matching", {})
    mask_cfg = flow_cfg.get("mask", {})

    loss_config = LFlowLossConfig(
        lambda_gen=float(loss_cfg.get("lambda_gen", 1.0)),
        lambda_prog=float(loss_cfg.get("lambda_prog", 0.1)),
        lambda_ord=float(loss_cfg.get("lambda_ord", 0.1)),
        generation_loss=str(loss_cfg.get("generation_loss", "mse")),
        progression_distance=str(prog_cfg.get("distance", "l2")),
        ordinal_margin=float(ord_cfg.get("margin", 0.1)),
        ordinal_mode=str(ord_cfg.get("mode", "pairwise_margin")),
        use_masked_region_only=bool(
            loss_cfg.get("use_masked_region_only", True)
        ),
    )

    return LFlowLoss(
        hidden_dim=hidden_dim,
        config=loss_config,
    )


# ---------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    batch_size = 4
    n_mels = 100
    frames = 128
    hidden_dim = 256

    predicted_vector_field = torch.randn(batch_size, n_mels, frames)
    target_vector_field = torch.randn(batch_size, n_mels, frames)

    mask = torch.zeros(batch_size, n_mels, frames)
    mask[:, :, 32:96] = 1.0

    z_prog = torch.randn(batch_size, hidden_dim)
    target_time = torch.tensor([0.0, 30.0, 60.0, 90.0])
    target_label = torch.tensor([0.0, 0.2, 0.4, 0.8])

    criterion = LFlowLoss(hidden_dim=hidden_dim)

    losses = criterion(
        predicted_vector_field=predicted_vector_field,
        target_vector_field=target_vector_field,
        z_prog=z_prog,
        target_time=target_time,
        target_label=target_label,
        mask=mask,
    )

    for key, value in losses.items():
        print(key, float(value))