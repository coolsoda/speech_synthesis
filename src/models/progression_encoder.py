"""

The progression encoder maps a subject's longitudinal history and a query time
to a query-specific latent progression representation z_prog.

Input:
    context_mels:
        Tensor of shape [B, T, F, N]
        B = batch size
        T = number of context sessions
        F = mel bins
        N = mel frames

    context_times:
        Tensor of shape [B, T]

    context_labels:
        Tensor of shape [B, T]

    query_time:
        Tensor of shape [B]

    context_mask:
        Tensor of shape [B, T]
        1 = valid context session
        0 = padded context session

Output:
    z_prog:
        Tensor of shape [B, hidden_dim]

This module is intentionally self-contained and does not depend on the generator.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for scalar time values.

    The input can be absolute timestamp, days from baseline, or relative time.
    In practice, relative time to the query is usually more stable:

        delta_t = context_time - query_time

    This layer maps scalar time values to a dense vector.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")

        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t:
            Tensor of shape [...].

        Returns
        -------
        emb:
            Tensor of shape [..., dim].
        """

        half_dim = self.dim // 2
        device = t.device
        dtype = t.dtype

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=dtype)
            / max(half_dim, 1)
        )

        args = t.unsqueeze(-1) * frequencies
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb


class ScalarEmbedding(nn.Module):
    """
    MLP embedding for scalar values such as normalized severity labels.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or output_dim

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape [...] or [..., 1].

        Returns
        -------
        Tensor of shape [..., output_dim].
        """

        if x.ndim == 0:
            x = x.view(1, 1)
        elif x.shape[-1:] != (1,):
            x = x.unsqueeze(-1)

        return self.net(x)


class ConvRecordingEncoder(nn.Module):
    """
    Compact recording-level acoustic encoder.

    This module converts a mel-spectrogram into a fixed-dimensional recording
    representation. It is intentionally lightweight, because the main temporal
    modeling happens in the Transformer progression encoder.

    Input:
        mel: [B, T, F, N] or [B, F, N]

    Output:
        recording_embedding:
            [B, T, recording_dim] or [B, recording_dim]
    """

    def __init__(
        self,
        n_mels: int = 100,
        recording_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.recording_dim = recording_dim

        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, recording_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=recording_dim),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        original_ndim = mel.ndim

        if original_ndim == 4:
            batch_size, num_sessions, n_mels, num_frames = mel.shape
            mel = mel.reshape(batch_size * num_sessions, n_mels, num_frames)
        elif original_ndim == 3:
            batch_size, n_mels, num_frames = mel.shape
            num_sessions = None
        else:
            raise ValueError(
                "Expected mel shape [B, T, F, N] or [B, F, N]. "
                f"Received shape {tuple(mel.shape)}."
            )

        if mel.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} mel bins, received {mel.shape[1]}."
            )

        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)

        if original_ndim == 4:
            x = x.reshape(batch_size, num_sessions, self.recording_dim)

        return x


# ---------------------------------------------------------------------
# Progression encoder
# ---------------------------------------------------------------------


class LongitudinalProgressionEncoder(nn.Module):
    """
    Transformer-based longitudinal progression encoder.

    The encoder builds one token per observed recording session and one query
    token for the target/query time. The Transformer encoder attends over the
    full subject history, and the final query token hidden state is used as
    z_prog.

    Each observed recording token contains:
        acoustic embedding
        relative time embedding
        severity/label embedding

    The query token contains:
        a learnable query embedding
        query-time embedding
        optional query severity embedding

    For training:
        query_time = target_time
        query_label = target_label

    For inference:
        query_time = synthetic intermediate timestamp
        query_label = soft interpolated label
    """

    def __init__(
        self,
        n_mels: int = 100,
        recording_dim: int = 256,
        time_dim: int = 64,
        label_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_label: bool = True,
        use_relative_time: bool = True,
        normalize_relative_time: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.n_mels = n_mels
        self.recording_dim = recording_dim
        self.time_dim = time_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.use_label = use_label
        self.use_relative_time = use_relative_time
        self.normalize_relative_time = normalize_relative_time

        self.recording_encoder = ConvRecordingEncoder(
            n_mels=n_mels,
            recording_dim=recording_dim,
            hidden_dim=max(128, recording_dim // 2),
            dropout=dropout,
        )

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        self.label_embedding = ScalarEmbedding(
            output_dim=label_dim,
            hidden_dim=label_dim,
            dropout=dropout,
        )

        token_input_dim = recording_dim + time_dim
        if use_label:
            token_input_dim += label_dim

        self.context_token_projection = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        query_input_dim = hidden_dim + time_dim
        if use_label:
            query_input_dim += label_dim

        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

        self.query_token_projection = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context_mels: torch.Tensor,
        context_times: torch.Tensor,
        context_labels: torch.Tensor,
        query_time: torch.Tensor,
        query_label: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        context_mels:
            Tensor of shape [B, T, F, N].
        context_times:
            Tensor of shape [B, T].
        context_labels:
            Tensor of shape [B, T].
        query_time:
            Tensor of shape [B].
        query_label:
            Optional tensor of shape [B].
        context_mask:
            Optional tensor of shape [B, T], where 1 means valid and 0 means padded.
        return_all_tokens:
            If True, return both z_prog and all Transformer output tokens.

        Returns
        -------
        z_prog:
            Tensor of shape [B, hidden_dim].
        """

        if context_mels.ndim != 4:
            raise ValueError(
                "context_mels must have shape [B, T, F, N]. "
                f"Received {tuple(context_mels.shape)}."
            )

        batch_size, num_context, n_mels, _ = context_mels.shape

        if n_mels != self.n_mels:
            raise ValueError(
                f"Expected context_mels with {self.n_mels} mel bins, got {n_mels}."
            )

        if context_times.shape != (batch_size, num_context):
            raise ValueError(
                "context_times must have shape [B, T]. "
                f"Expected {(batch_size, num_context)}, got {tuple(context_times.shape)}."
            )

        if context_labels.shape != (batch_size, num_context):
            raise ValueError(
                "context_labels must have shape [B, T]. "
                f"Expected {(batch_size, num_context)}, got {tuple(context_labels.shape)}."
            )

        if query_time.shape != (batch_size,):
            query_time = query_time.view(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                batch_size,
                num_context,
                device=context_mels.device,
                dtype=torch.float32,
            )
        else:
            context_mask = context_mask.to(device=context_mels.device, dtype=torch.float32)

        context_acoustic = self.recording_encoder(context_mels)

        relative_times = self._compute_relative_times(
            context_times=context_times,
            query_time=query_time,
            context_mask=context_mask,
        )

        context_time_emb = self.time_embedding(relative_times)

        token_parts = [context_acoustic, context_time_emb]

        if self.use_label:
            context_label_emb = self.label_embedding(context_labels)
            token_parts.append(context_label_emb)

        context_tokens = torch.cat(token_parts, dim=-1)
        context_tokens = self.context_token_projection(context_tokens)

        query_tokens = self._build_query_token(
            query_time=query_time,
            query_label=query_label,
            batch_size=batch_size,
            dtype=context_tokens.dtype,
            device=context_tokens.device,
        )

        tokens = torch.cat([context_tokens, query_tokens], dim=1)

        # Transformer expects True for positions that should be ignored.
        query_mask = torch.ones(
            batch_size,
            1,
            device=context_mask.device,
            dtype=context_mask.dtype,
        )
        full_valid_mask = torch.cat([context_mask, query_mask], dim=1)
        src_key_padding_mask = full_valid_mask == 0

        encoded = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )

        query_output = encoded[:, -1, :]
        z_prog = self.output_norm(query_output)

        if return_all_tokens:
            return z_prog, encoded

        return z_prog

    def _compute_relative_times(
        self,
        context_times: torch.Tensor,
        query_time: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relative time values for context sessions.

        By default:
            delta_t = context_time - query_time

        If normalize_relative_time=True, divide by the maximum absolute valid
        delta per subject to reduce scale variation across datasets.
        """

        if self.use_relative_time:
            relative_times = context_times - query_time.unsqueeze(1)
        else:
            relative_times = context_times

        if not self.normalize_relative_time:
            return relative_times

        valid_abs = relative_times.abs() * context_mask
        max_abs = valid_abs.max(dim=1, keepdim=True).values.clamp_min(1.0)
        relative_times = relative_times / max_abs

        # Make padded positions harmless.
        relative_times = relative_times * context_mask

        return relative_times

    def _build_query_token(
        self,
        query_time: torch.Tensor,
        query_label: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the query-time token.
        """

        query_base = self.query_token.expand(batch_size, 1, -1)

        # The query time relative to itself is zero. This makes the query token
        # stable across absolute timestamp scales.
        query_relative_time = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )
        query_time_emb = self.time_embedding(query_relative_time)

        query_parts = [query_base, query_time_emb]

        if self.use_label:
            if query_label is None:
                query_label = torch.zeros(
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                query_label = query_label.to(device=device, dtype=dtype).view(batch_size)

            query_label_emb = self.label_embedding(query_label).unsqueeze(1)
            query_parts.append(query_label_emb)

        query_input = torch.cat(query_parts, dim=-1)
        query_token = self.query_token_projection(query_input)

        return query_token


# ---------------------------------------------------------------------
# Optional lightweight encoder without audio context
# ---------------------------------------------------------------------


class MetadataOnlyProgressionEncoder(nn.Module):
    """
    A lightweight progression encoder that uses only time and label history.

    This is useful for:
    - debugging;
    - ablation without acoustic progression modeling;
    - fast unit tests.

    It has the same output interface as LongitudinalProgressionEncoder.
    """

    def __init__(
        self,
        time_dim: int = 64,
        label_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_label: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.label_embedding = ScalarEmbedding(label_dim, dropout=dropout)
        self.use_label = use_label
        self.hidden_dim = hidden_dim

        token_input_dim = time_dim + (label_dim if use_label else 0)

        self.context_projection = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

        query_input_dim = hidden_dim + time_dim + (label_dim if use_label else 0)

        self.query_projection = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context_times: torch.Tensor,
        context_labels: torch.Tensor,
        query_time: torch.Tensor,
        query_label: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        **_: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_context = context_times.shape
        device = context_times.device
        dtype = context_times.dtype

        if context_mask is None:
            context_mask = torch.ones(
                batch_size,
                num_context,
                device=device,
                dtype=dtype,
            )
        else:
            context_mask = context_mask.to(device=device, dtype=dtype)

        relative_times = context_times - query_time.view(batch_size, 1)
        max_abs = (relative_times.abs() * context_mask).max(
            dim=1,
            keepdim=True,
        ).values.clamp_min(1.0)

        relative_times = (relative_times / max_abs) * context_mask
        time_emb = self.time_embedding(relative_times)

        parts = [time_emb]

        if self.use_label:
            label_emb = self.label_embedding(context_labels)
            parts.append(label_emb)

        context_tokens = torch.cat(parts, dim=-1)
        context_tokens = self.context_projection(context_tokens)

        query_base = self.query_token.expand(batch_size, 1, -1)
        query_time_emb = self.time_embedding(
            torch.zeros(batch_size, 1, device=device, dtype=dtype)
        )

        query_parts = [query_base, query_time_emb]

        if self.use_label:
            if query_label is None:
                query_label = torch.zeros(batch_size, device=device, dtype=dtype)
            query_label_emb = self.label_embedding(query_label.view(batch_size)).unsqueeze(1)
            query_parts.append(query_label_emb)

        query_token = self.query_projection(torch.cat(query_parts, dim=-1))

        tokens = torch.cat([context_tokens, query_token], dim=1)

        query_mask = torch.ones(batch_size, 1, device=device, dtype=dtype)
        valid_mask = torch.cat([context_mask, query_mask], dim=1)
        src_key_padding_mask = valid_mask == 0

        encoded = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )

        z_prog = self.output_norm(encoded[:, -1, :])

        if return_all_tokens:
            return z_prog, encoded

        return z_prog


# ---------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------


def build_progression_encoder_from_config(config: dict) -> nn.Module:
    """
    Build progression encoder from loaded YAML config.

    Expected config section:

    model:
      progression_encoder:
        recording_dim: 256
        time_dim: 64
        label_dim: 64
        hidden_dim: 256
        num_layers: 4
        num_heads: 4
        dropout: 0.1
        pooling: "query_token"
    """

    model_cfg = config.get("model", {})
    enc_cfg = model_cfg.get("progression_encoder", {})

    encoder_type = enc_cfg.get("type", "acoustic")

    common_kwargs = {
        "time_dim": int(enc_cfg.get("time_dim", 64)),
        "label_dim": int(enc_cfg.get("label_dim", 64)),
        "hidden_dim": int(enc_cfg.get("hidden_dim", 256)),
        "num_layers": int(enc_cfg.get("num_layers", 4)),
        "num_heads": int(enc_cfg.get("num_heads", 4)),
        "dropout": float(enc_cfg.get("dropout", 0.1)),
        "ff_mult": int(enc_cfg.get("ff_mult", 4)),
        "use_label": bool(enc_cfg.get("use_label", True)),
    }

    if encoder_type == "metadata_only":
        return MetadataOnlyProgressionEncoder(**common_kwargs)

    if encoder_type != "acoustic":
        raise ValueError(
            f"Unknown progression encoder type: {encoder_type}. "
            "Expected 'acoustic' or 'metadata_only'."
        )

    preprocessing_cfg = config.get("preprocessing", {})
    mel_cfg = preprocessing_cfg.get("mel", {})

    return LongitudinalProgressionEncoder(
        n_mels=int(mel_cfg.get("n_mels", 100)),
        recording_dim=int(enc_cfg.get("recording_dim", 256)),
        normalize_relative_time=bool(enc_cfg.get("normalize_relative_time", True)),
        use_relative_time=bool(enc_cfg.get("use_relative_time", True)),
        **common_kwargs,
    )


# ---------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    batch_size = 2
    num_context = 4
    n_mels = 100
    num_frames = 128

    context_mels = torch.randn(batch_size, num_context, n_mels, num_frames)
    context_times = torch.tensor(
        [
            [0.0, 30.0, 60.0, 90.0],
            [0.0, 800.0, 1600.0, 2400.0],
        ]
    )
    context_labels = torch.tensor(
        [
            [0.0, 0.2, 0.3, 0.5],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    query_time = torch.tensor([45.0, 1200.0])
    query_label = torch.tensor([0.25, 0.5])
    context_mask = torch.ones(batch_size, num_context)

    encoder = LongitudinalProgressionEncoder(
        n_mels=n_mels,
        recording_dim=256,
        time_dim=64,
        label_dim=64,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
    )

    z_prog = encoder(
        context_mels=context_mels,
        context_times=context_times,
        context_labels=context_labels,
        query_time=query_time,
        query_label=query_label,
        context_mask=context_mask,
    )

    print("z_prog shape:", z_prog.shape)