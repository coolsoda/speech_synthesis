"""
Conditional flow-based speech generator for L-Flow.

The generator receives:
    noisy_mel:
        Mel point on the flow path, shape [B, F, N]

    observed_mel:
        Partially observed / masked target mel, shape [B, F, N]

    mask:
        Binary mask, shape [B, F, N], where 1 means masked region

    text_ids:
        Character-level text tokens, shape [B, N_text]

    reference_mel:
        Reference speech mel, shape [B, F, N_ref]

    z_prog:
        Query-specific progression state, shape [B, D_prog]

    sigma:
        Flow step, shape [B]

    severity:
        Normalized severity/progression label, shape [B]

Output:
    predicted_vector_field:
        Tensor of shape [B, F, N]

"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic embeddings
# ---------------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar values such as flow step sigma.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")

        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape [B] or arbitrary shape.

        Returns
        -------
        Tensor with shape [..., dim].
        """

        half_dim = self.dim // 2
        device = x.device
        dtype = x.dtype

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=dtype)
            / max(half_dim, 1)
        )

        args = x.unsqueeze(-1) * frequencies
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ScalarConditionEmbedding(nn.Module):
    """
    Small MLP for scalar conditions such as normalized severity labels.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or dim

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x.view(1, 1)
        elif x.shape[-1:] != (1,):
            x = x.unsqueeze(-1)

        return self.net(x)


# ---------------------------------------------------------------------
# Text and reference encoders
# ---------------------------------------------------------------------


class TextConditionEncoder(nn.Module):
    """
    Character-level text condition encoder.

    Input:
        text_ids: [B, N_text]

    Output:
        text_features: [B, N_text, hidden_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.max_length = max_length

    def forward(
        self,
        text_ids: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        if text_ids.ndim != 2:
            raise ValueError(
                f"Expected text_ids shape [B, N], got {tuple(text_ids.shape)}."
            )

        batch_size, seq_len = text_ids.shape

        if seq_len > self.max_length:
            text_ids = text_ids[:, : self.max_length]
            seq_len = self.max_length

        positions = torch.arange(seq_len, device=text_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embedding(text_ids) + self.position_embedding(positions)

        key_padding_mask = text_ids == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if target_length is not None and seq_len != target_length:
            x = interpolate_sequence(x, target_length=target_length)

        return x


class ReferenceSpeechEncoder(nn.Module):
    """
    Lightweight encoder for reference speech mel-spectrograms.

    Input:
        reference_mel: [B, F, N]

    Output:
        reference_features: [B, N, hidden_dim]
    """

    def __init__(
        self,
        n_mels: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels

        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        reference_mel: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        if reference_mel.ndim != 3:
            raise ValueError(
                "Expected reference_mel shape [B, F, N], "
                f"got {tuple(reference_mel.shape)}."
            )

        if reference_mel.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} mel bins, got {reference_mel.shape[1]}."
            )

        x = self.net(reference_mel)  # [B, hidden_dim, N]
        x = x.transpose(1, 2)       # [B, N, hidden_dim]

        if target_length is not None and x.shape[1] != target_length:
            x = interpolate_sequence(x, target_length=target_length)

        return x


# ---------------------------------------------------------------------
# DiT-style conditioning blocks
# ---------------------------------------------------------------------


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive LayerNorm.

    Given a conditioning vector c, produce scale and shift values:

        AdaLN(x, c) = LN(x) * (1 + scale(c)) + shift(c)

    Zero initialization makes the block start close to a standard Transformer.
    """

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim * 2)

        if zero_init:
            nn.init.zeros_(self.condition_proj.weight)
            nn.init.zeros_(self.condition_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_proj(condition).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        return self.norm(x) * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """
    Diffusion Transformer-style block with adaptive LayerNorm conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        condition_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.adaln_attn = AdaptiveLayerNorm(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            zero_init=True,
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.adaln_ff = AdaptiveLayerNorm(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            zero_init=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_input = self.adaln_attn(x, condition)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_output

        ff_input = self.adaln_ff(x, condition)
        x = x + self.ff(ff_input)

        return x


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def interpolate_sequence(
    x: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """
    Interpolate sequence features along the time dimension.

    Input:
        x: [B, N, D]

    Output:
        x: [B, target_length, D]
    """

    if x.ndim != 3:
        raise ValueError(f"Expected x shape [B, N, D], got {tuple(x.shape)}.")

    x = x.transpose(1, 2)  # [B, D, N]
    x = F.interpolate(
        x,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    x = x.transpose(1, 2)  # [B, target_length, D]

    return x


def make_mel_padding_mask(
    mel: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Infer a time padding mask from mel tensors.

    Returns:
        key_padding_mask: [B, N], where True means padded.

    This is a fallback. If your training later uses explicit frame lengths,
    replace this with a length-based mask.
    """

    # mel: [B, F, N]
    energy = mel.abs().sum(dim=1)
    return energy <= eps


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------


class ConditionalFlowGenerator(nn.Module):
    """
    Conditional flow-matching speech generator.

    The model predicts the vector field over mel-spectrogram space.
    """

    def __init__(
        self,
        n_mels: int = 100,
        vocab_size: int = 80,
        hidden_dim: int = 512,
        progression_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_length: int = 4096,
        use_text_condition: bool = True,
        use_reference_speech: bool = True,
        use_progression_condition: bool = True,
        use_severity_condition: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.progression_dim = progression_dim

        self.use_text_condition = use_text_condition
        self.use_reference_speech = use_reference_speech
        self.use_progression_condition = use_progression_condition
        self.use_severity_condition = use_severity_condition

        # Input contains:
        #   noisy mel
        #   observed mel
        #   mask
        input_channels = n_mels * 2 + n_mels

        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.max_length = max_length

        if use_text_condition:
            self.text_encoder = TextConditionEncoder(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=2,
                num_heads=max(1, min(num_heads, 4)),
                dropout=dropout,
                max_length=max_length,
            )
            self.text_projection = nn.Linear(hidden_dim, hidden_dim)

        if use_reference_speech:
            self.reference_encoder = ReferenceSpeechEncoder(
                n_mels=n_mels,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.reference_projection = nn.Linear(hidden_dim, hidden_dim)

        self.sigma_embedding = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_progression_condition:
            self.progression_projection = nn.Sequential(
                nn.Linear(progression_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if use_severity_condition:
            self.severity_embedding = ScalarConditionEmbedding(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        # This global condition is passed to AdaLN in every DiT block.
        condition_dim = hidden_dim

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    condition_dim=condition_dim,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_mels),
        )

    def forward(
        self,
        noisy_mel: torch.Tensor,
        observed_mel: torch.Tensor,
        mask: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        z_prog: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        severity: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        noisy_mel:
            [B, F, N]
        observed_mel:
            [B, F, N]
        mask:
            [B, F, N]
        text_ids:
            [B, N_text]
        reference_mel:
            [B, F, N_ref]
        z_prog:
            [B, progression_dim]
        sigma:
            [B]
        severity:
            [B]

        Returns
        -------
        predicted_vector_field:
            [B, F, N]
        """

        self._validate_mel_inputs(noisy_mel, observed_mel, mask)

        batch_size, n_mels, num_frames = noisy_mel.shape

        if num_frames > self.max_length:
            raise ValueError(
                f"Number of frames {num_frames} exceeds max_length={self.max_length}."
            )

        if sigma is None:
            sigma = torch.zeros(
                batch_size,
                device=noisy_mel.device,
                dtype=noisy_mel.dtype,
            )

        sigma = sigma.view(batch_size).to(
            device=noisy_mel.device,
            dtype=noisy_mel.dtype,
        )

        x = torch.cat([noisy_mel, observed_mel, mask], dim=1)
        x = self.input_projection(x)       # [B, D, N]
        x = x.transpose(1, 2)              # [B, N, D]

        positions = torch.arange(num_frames, device=noisy_mel.device)
        positions = positions.unsqueeze(0).expand(batch_size, num_frames)
        x = x + self.position_embedding(positions)

        global_condition = self.sigma_embedding(sigma)

        if self.use_text_condition:
            if text_ids is None:
                raise ValueError("text_ids must be provided when use_text_condition=True.")

            text_features = self.text_encoder(
                text_ids=text_ids,
                target_length=num_frames,
            )
            x = x + self.text_projection(text_features)

        if self.use_reference_speech:
            if reference_mel is None:
                raise ValueError(
                    "reference_mel must be provided when use_reference_speech=True."
                )

            reference_features = self.reference_encoder(
                reference_mel=reference_mel,
                target_length=num_frames,
            )
            x = x + self.reference_projection(reference_features)

        if self.use_progression_condition:
            if z_prog is None:
                raise ValueError(
                    "z_prog must be provided when use_progression_condition=True."
                )

            z_prog = z_prog.to(device=noisy_mel.device, dtype=noisy_mel.dtype)
            global_condition = global_condition + self.progression_projection(z_prog)

        if self.use_severity_condition:
            if severity is None:
                raise ValueError(
                    "severity must be provided when use_severity_condition=True."
                )

            severity = severity.view(batch_size).to(
                device=noisy_mel.device,
                dtype=noisy_mel.dtype,
            )
            global_condition = global_condition + self.severity_embedding(severity)

        if key_padding_mask is None:
            key_padding_mask = make_mel_padding_mask(observed_mel)

        for block in self.blocks:
            x = block(
                x,
                condition=global_condition,
                key_padding_mask=key_padding_mask,
            )

        x = self.final_norm(x)
        vector_field = self.output_projection(x)  # [B, N, F]
        vector_field = vector_field.transpose(1, 2)  # [B, F, N]

        return vector_field

    def _validate_mel_inputs(
        self,
        noisy_mel: torch.Tensor,
        observed_mel: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if noisy_mel.ndim != 3:
            raise ValueError(
                f"Expected noisy_mel shape [B, F, N], got {tuple(noisy_mel.shape)}."
            )

        if observed_mel.shape != noisy_mel.shape:
            raise ValueError(
                "observed_mel must have the same shape as noisy_mel. "
                f"Got {tuple(observed_mel.shape)} and {tuple(noisy_mel.shape)}."
            )

        if mask.shape != noisy_mel.shape:
            raise ValueError(
                "mask must have the same shape as noisy_mel. "
                f"Got {tuple(mask.shape)} and {tuple(noisy_mel.shape)}."
            )

        if noisy_mel.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} mel bins, got {noisy_mel.shape[1]}."
            )


# ---------------------------------------------------------------------
# Full L-Flow model wrapper
# ---------------------------------------------------------------------


class LFlowModel(nn.Module):
    """
    Full L-Flow model wrapper.

    This combines:
        1. Longitudinal progression encoder.
        2. Conditional flow generator.

    The progression encoder is passed in from progression_encoder.py so this
    file does not duplicate that implementation.
    """

    def __init__(
        self,
        progression_encoder: nn.Module,
        generator: ConditionalFlowGenerator,
    ) -> None:
        super().__init__()

        self.progression_encoder = progression_encoder
        self.generator = generator

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Expected batch fields:
            context_mels
            context_times
            context_labels
            context_mask
            target_time
            target_label
            noisy_mel
            observed_mel
            mask
            text_ids
            reference_mel
            sigma
        """

        z_prog = self.progression_encoder(
            context_mels=batch["context_mels"],
            context_times=batch["context_times"],
            context_labels=batch["context_labels"],
            query_time=batch["target_time"],
            query_label=batch.get("target_label"),
            context_mask=batch.get("context_mask"),
        )

        predicted_vector_field = self.generator(
            noisy_mel=batch["noisy_mel"],
            observed_mel=batch["observed_mel"],
            mask=batch["mask"],
            text_ids=batch.get("text_ids"),
            reference_mel=batch.get("reference_mel"),
            z_prog=z_prog,
            sigma=batch.get("sigma"),
            severity=batch.get("target_label"),
        )

        return {
            "predicted_vector_field": predicted_vector_field,
            "z_prog": z_prog,
        }

    @torch.no_grad()
    def generate_mel(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 32,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Generate mel-spectrograms by solving the learned ODE.

        This method starts from Gaussian noise and integrates:

            d psi_sigma / d sigma = v_theta(psi_sigma | conditions)

        from sigma = 0 to sigma = 1.

        Parameters
        ----------
        batch:
            Inference batch from lflow_inference_collate_fn.
        num_steps:
            Number of ODE integration steps.
        solver:
            Currently supports "euler" and "heun".

        Returns
        -------
        generated_mel:
            Tensor of shape [B, F, N].
        """

        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be either 'euler' or 'heun'.")

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        reference_mel = batch["reference_mel"]
        device = reference_mel.device
        dtype = reference_mel.dtype

        batch_size, n_mels, num_frames = reference_mel.shape

        z_prog = self.progression_encoder(
            context_mels=batch["context_mels"],
            context_times=batch["context_times"],
            context_labels=batch["context_labels"],
            query_time=batch["query_timestamp"],
            query_label=batch.get("soft_label"),
            context_mask=batch.get("context_mask"),
        )

        psi = torch.randn(
            batch_size,
            n_mels,
            num_frames,
            device=device,
            dtype=dtype,
        )

        observed_mel = batch["observed_mel"]
        mask = batch["mask"]
        text_ids = batch.get("text_ids")
        severity = batch.get("soft_label")

        step_size = 1.0 / float(num_steps)

        for step in range(num_steps):
            sigma_value = step / float(num_steps)
            sigma = torch.full(
                (batch_size,),
                fill_value=sigma_value,
                device=device,
                dtype=dtype,
            )

            if solver == "euler":
                vector_field = self.generator(
                    noisy_mel=psi,
                    observed_mel=observed_mel,
                    mask=mask,
                    text_ids=text_ids,
                    reference_mel=reference_mel,
                    z_prog=z_prog,
                    sigma=sigma,
                    severity=severity,
                )
                psi = psi + step_size * vector_field

            else:
                # Heun / improved Euler step.
                vector_field_1 = self.generator(
                    noisy_mel=psi,
                    observed_mel=observed_mel,
                    mask=mask,
                    text_ids=text_ids,
                    reference_mel=reference_mel,
                    z_prog=z_prog,
                    sigma=sigma,
                    severity=severity,
                )

                psi_pred = psi + step_size * vector_field_1

                sigma_next = torch.full(
                    (batch_size,),
                    fill_value=(step + 1) / float(num_steps),
                    device=device,
                    dtype=dtype,
                )

                vector_field_2 = self.generator(
                    noisy_mel=psi_pred,
                    observed_mel=observed_mel,
                    mask=mask,
                    text_ids=text_ids,
                    reference_mel=reference_mel,
                    z_prog=z_prog,
                    sigma=sigma_next,
                    severity=severity,
                )

                psi = psi + 0.5 * step_size * (vector_field_1 + vector_field_2)

        return psi


# ---------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------


def build_generator_from_config(
    config: dict,
    vocab_size: int,
) -> ConditionalFlowGenerator:
    """
    Build ConditionalFlowGenerator from loaded YAML config.
    """

    preprocessing_cfg = config.get("preprocessing", {})
    mel_cfg = preprocessing_cfg.get("mel", {})

    model_cfg = config.get("model", {})
    gen_cfg = model_cfg.get("generator", {})
    prog_cfg = model_cfg.get("progression_encoder", {})

    n_mels = int(gen_cfg.get("mel_dim", mel_cfg.get("n_mels", 100)))

    return ConditionalFlowGenerator(
        n_mels=n_mels,
        vocab_size=vocab_size,
        hidden_dim=int(gen_cfg.get("hidden_dim", 512)),
        progression_dim=int(prog_cfg.get("hidden_dim", 256)),
        num_layers=int(gen_cfg.get("num_layers", 12)),
        num_heads=int(gen_cfg.get("num_heads", 8)),
        ff_mult=int(gen_cfg.get("ff_mult", 4)),
        dropout=float(gen_cfg.get("dropout", 0.1)),
        max_length=int(gen_cfg.get("max_length", 4096)),
        use_text_condition=bool(gen_cfg.get("use_text_condition", True)),
        use_reference_speech=bool(gen_cfg.get("use_reference_speech", True)),
        use_progression_condition=bool(gen_cfg.get("use_progression_condition", True)),
        use_severity_condition=bool(gen_cfg.get("use_severity_condition", True)),
    )


def build_lflow_model_from_config(
    config: dict,
    vocab_size: int,
) -> LFlowModel:
    """
    Build the full L-Flow model from config.

    This imports the progression encoder factory here to avoid circular imports.
    """

    from .progression_encoder import build_progression_encoder_from_config

    progression_encoder = build_progression_encoder_from_config(config)
    generator = build_generator_from_config(config, vocab_size=vocab_size)

    return LFlowModel(
        progression_encoder=progression_encoder,
        generator=generator,
    )


# ---------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    from .progression_encoder import LongitudinalProgressionEncoder

    batch_size = 2
    num_context = 3
    n_mels = 100
    num_frames = 128
    vocab_size = 80

    progression_encoder = LongitudinalProgressionEncoder(
        n_mels=n_mels,
        recording_dim=256,
        time_dim=64,
        label_dim=64,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
    )

    generator = ConditionalFlowGenerator(
        n_mels=n_mels,
        vocab_size=vocab_size,
        hidden_dim=512,
        progression_dim=256,
        num_layers=4,
        num_heads=8,
    )

    model = LFlowModel(
        progression_encoder=progression_encoder,
        generator=generator,
    )

    batch = {
        "context_mels": torch.randn(batch_size, num_context, n_mels, num_frames),
        "context_times": torch.tensor(
            [
                [0.0, 30.0, 60.0],
                [0.0, 800.0, 1600.0],
            ]
        ),
        "context_labels": torch.tensor(
            [
                [0.0, 0.2, 0.4],
                [0.0, 0.0, 1.0],
            ]
        ),
        "context_mask": torch.ones(batch_size, num_context),

        "target_time": torch.tensor([45.0, 1200.0]),
        "target_label": torch.tensor([0.3, 0.5]),

        "noisy_mel": torch.randn(batch_size, n_mels, num_frames),
        "observed_mel": torch.randn(batch_size, n_mels, num_frames),
        "mask": torch.ones(batch_size, n_mels, num_frames),

        "text_ids": torch.randint(0, vocab_size, (batch_size, num_frames)),
        "reference_mel": torch.randn(batch_size, n_mels, num_frames),
        "sigma": torch.rand(batch_size),
    }

    output = model(batch)

    print("predicted_vector_field:", output["predicted_vector_field"].shape)
    print("z_prog:", output["z_prog"].shape)