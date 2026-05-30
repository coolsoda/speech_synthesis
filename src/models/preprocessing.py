"""
Audio, text, mel-spectrogram, and masking utilities for L-Flow.

The design follows the L-Flow training setup, where the model receives a
partially observed target mel-spectrogram, text condition, reference speech,
longitudinal context, and progression labels. The paper defines masked speech
reconstruction using a binary mask over the target mel-spectrogram and pads the
text condition with filler tokens to match the mel length. :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio


PathLike = Union[str, Path]


@dataclass
class MelConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    log_mel: bool = True
    power: float = 2.0


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    max_audio_seconds: Optional[float] = 30.0
    min_audio_seconds: float = 1.0
    trim_silence: bool = True
    silence_top_db: int = 40
    normalize_waveform: bool = True


@dataclass
class TextConfig:
    lowercase: bool = True
    use_char_level: bool = True
    filler_token: str = "<F>"
    pad_to_mel_length: bool = True


@dataclass
class MaskConfig:
    strategy: str = "random_span"
    min_mask_ratio: float = 0.3
    max_mask_ratio: float = 0.8


# ---------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------


def load_audio(
    audio_path: PathLike,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and optionally resample it.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    target_sample_rate:
        Desired sampling rate.
    mono:
        If True, convert multi-channel audio to mono.

    Returns
    -------
    waveform:
        Tensor with shape [1, num_samples] if mono=True.
    sample_rate:
        Final sampling rate.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = torchaudio.load(str(audio_path))

    if mono and waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate

    return waveform, sample_rate


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Peak-normalize a waveform to [-1, 1].
    """

    max_abs = waveform.abs().max()
    if max_abs < eps:
        return waveform
    return waveform / max_abs.clamp_min(eps)


def trim_silence(
    waveform: torch.Tensor,
    top_db: int = 40,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Trim leading and trailing silence using a simple energy threshold.

    This avoids adding a librosa dependency. The implementation computes
    frame-wise RMS energy and keeps the region whose energy is within top_db
    of the maximum frame energy.
    """

    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [channels, samples].")

    if waveform.shape[-1] < frame_length:
        return waveform

    mono = waveform.mean(dim=0)

    frames = mono.unfold(dimension=0, size=frame_length, step=hop_length)
    rms = torch.sqrt(torch.mean(frames**2, dim=1) + 1e-10)

    max_rms = rms.max()
    if max_rms <= 0:
        return waveform

    threshold = max_rms * (10.0 ** (-top_db / 20.0))
    active = torch.nonzero(rms >= threshold, as_tuple=False).flatten()

    if active.numel() == 0:
        return waveform

    start_frame = int(active[0].item())
    end_frame = int(active[-1].item())

    start_sample = max(0, start_frame * hop_length)
    end_sample = min(
        waveform.shape[-1],
        end_frame * hop_length + frame_length,
    )

    return waveform[:, start_sample:end_sample]


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    max_audio_seconds: Optional[float] = 30.0,
    min_audio_seconds: float = 1.0,
    random_crop: bool = False,
) -> torch.Tensor:
    """
    Crop or pad waveform to satisfy duration constraints.

    For training, random cropping can be enabled. For validation/inference,
    deterministic cropping from the beginning is preferred.
    """

    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [channels, samples].")

    num_samples = waveform.shape[-1]
    min_samples = int(min_audio_seconds * sample_rate)

    if num_samples < min_samples:
        pad_amount = min_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        num_samples = waveform.shape[-1]

    if max_audio_seconds is None:
        return waveform

    max_samples = int(max_audio_seconds * sample_rate)

    if num_samples <= max_samples:
        return waveform

    if random_crop:
        start = random.randint(0, num_samples - max_samples)
    else:
        start = 0

    end = start + max_samples
    return waveform[:, start:end]


# ---------------------------------------------------------------------
# Mel-spectrogram utilities
# ---------------------------------------------------------------------


class MelSpectrogramExtractor:
    """
    Wrapper for mel-spectrogram extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mel_config: Optional[MelConfig] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mel_config = mel_config or MelConfig()

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.mel_config.n_fft,
            hop_length=self.mel_config.hop_length,
            win_length=self.mel_config.win_length,
            n_mels=self.mel_config.n_mels,
            f_min=self.mel_config.f_min,
            f_max=self.mel_config.f_max,
            power=self.mel_config.power,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform into mel-spectrogram.

        Returns
        -------
        mel:
            Tensor with shape [n_mels, frames].
        """

        mel = self.transform(waveform)

        # torchaudio returns [channels, n_mels, frames].
        if mel.ndim == 3:
            mel = mel.mean(dim=0)

        if self.mel_config.log_mel:
            mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


def pad_or_crop_mel(
    mel: torch.Tensor,
    target_frames: int,
    random_crop: bool = False,
) -> torch.Tensor:
    """
    Pad or crop mel-spectrogram to a fixed number of frames.

    Parameters
    ----------
    mel:
        Tensor with shape [n_mels, frames].
    target_frames:
        Desired number of frames.
    """

    if mel.ndim != 2:
        raise ValueError("Expected mel shape [n_mels, frames].")

    n_mels, frames = mel.shape

    if frames == target_frames:
        return mel

    if frames < target_frames:
        pad_amount = target_frames - frames
        return torch.nn.functional.pad(mel, (0, pad_amount))

    if random_crop:
        start = random.randint(0, frames - target_frames)
    else:
        start = 0

    return mel[:, start : start + target_frames]


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------


class CharacterTokenizer:
    """
    Lightweight character-level tokenizer.

    This tokenizer is intentionally simple and self-contained. It supports
    lowercase English text, punctuation, whitespace, unknown characters, padding,
    and a filler token used to align text tokens with mel frames.
    """

    def __init__(
        self,
        lowercase: bool = True,
        filler_token: str = "<F>",
        extra_chars: Optional[Iterable[str]] = None,
    ) -> None:
        self.lowercase = lowercase
        self.filler_token = filler_token

        base_chars = list(string.ascii_lowercase)
        base_chars += list(string.digits)
        base_chars += list(" .,!?;:'\"-_/()[]{}")
        base_chars += ["\n", "\t"]

        if extra_chars is not None:
            base_chars += list(extra_chars)

        # Preserve order while removing duplicates.
        seen = set()
        vocab_chars = []
        for ch in base_chars:
            if ch not in seen:
                seen.add(ch)
                vocab_chars.append(ch)

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.tokens = [self.pad_token, self.unk_token, self.filler_token] + vocab_chars
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.filler_id = self.token_to_id[self.filler_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def normalize_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        if self.lowercase:
            text = text.lower()
        return text

    def encode(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        return [self.token_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Sequence[int], remove_special: bool = True) -> str:
        chars = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if remove_special and token in {
                self.pad_token,
                self.unk_token,
                self.filler_token,
            }:
                continue
            chars.append(token)
        return "".join(chars)

    def pad_to_length(
        self,
        token_ids: Sequence[int],
        target_length: int,
        pad_with_filler: bool = True,
    ) -> torch.Tensor:
        token_ids = list(token_ids)

        if len(token_ids) >= target_length:
            token_ids = token_ids[:target_length]
        else:
            pad_id = self.filler_id if pad_with_filler else self.pad_id
            token_ids = token_ids + [pad_id] * (target_length - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------


def create_random_span_mask(
    num_frames: int,
    min_mask_ratio: float = 0.3,
    max_mask_ratio: float = 0.8,
) -> torch.Tensor:
    """
    Create a 1D random contiguous span mask over time frames.

    Returns
    -------
    mask:
        Tensor with shape [num_frames], where 1 means masked and 0 means visible.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if not (0.0 <= min_mask_ratio <= max_mask_ratio <= 1.0):
        raise ValueError(
            "Mask ratios must satisfy 0 <= min_mask_ratio <= max_mask_ratio <= 1."
        )

    ratio = random.uniform(min_mask_ratio, max_mask_ratio)
    mask_length = max(1, int(round(num_frames * ratio)))
    mask_length = min(mask_length, num_frames)

    start = random.randint(0, num_frames - mask_length)
    end = start + mask_length

    mask = torch.zeros(num_frames, dtype=torch.float32)
    mask[start:end] = 1.0
    return mask


def create_random_mask(
    num_frames: int,
    min_mask_ratio: float = 0.3,
    max_mask_ratio: float = 0.8,
) -> torch.Tensor:
    """
    Create a random non-contiguous mask over time frames.
    """

    ratio = random.uniform(min_mask_ratio, max_mask_ratio)
    num_masked = max(1, int(round(num_frames * ratio)))
    num_masked = min(num_masked, num_frames)

    indices = torch.randperm(num_frames)[:num_masked]
    mask = torch.zeros(num_frames, dtype=torch.float32)
    mask[indices] = 1.0
    return mask


def create_full_mask(num_frames: int) -> torch.Tensor:
    """
    Create a full mask for inference-time full-sample generation.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    return torch.ones(num_frames, dtype=torch.float32)


def expand_time_mask(mask: torch.Tensor, n_mels: int) -> torch.Tensor:
    """
    Expand a time-frame mask to mel shape [n_mels, frames].
    """

    if mask.ndim != 1:
        raise ValueError("Expected mask shape [frames].")
    return mask.unsqueeze(0).expand(n_mels, -1)


def apply_mel_mask(mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mask to mel-spectrogram.

    In this codebase, mask value 1 means masked region.
    Therefore, the observed mel is:

        observed = (1 - mask) * mel
    """

    if mel.shape != mask.shape:
        raise ValueError(
            f"mel and mask must have the same shape. "
            f"Got mel={mel.shape}, mask={mask.shape}."
        )

    return (1.0 - mask) * mel


# ---------------------------------------------------------------------
# Flow-matching utilities
# ---------------------------------------------------------------------


def sample_flow_step(batch_size: int = 1) -> torch.Tensor:
    """
    Sample sigma from Uniform(0, 1).
    """

    return torch.rand(batch_size, dtype=torch.float32)


def create_noisy_path(
    target_mel: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create the linear flow-matching path between noise and target mel.

    psi_sigma = (1 - sigma) * epsilon + sigma * target_mel

    Returns
    -------
    psi:
        Noisy mel at flow step sigma.
    target_vector_field:
        The vector field target, target_mel - epsilon.
    """

    if noise is None:
        noise = torch.randn_like(target_mel)

    if isinstance(sigma, float):
        sigma_tensor = torch.tensor(sigma, dtype=target_mel.dtype, device=target_mel.device)
    else:
        sigma_tensor = sigma.to(dtype=target_mel.dtype, device=target_mel.device)

    while sigma_tensor.ndim < target_mel.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)

    psi = (1.0 - sigma_tensor) * noise + sigma_tensor * target_mel
    target_vector_field = target_mel - noise

    return psi, target_vector_field


# ---------------------------------------------------------------------
# High-level preprocessing class
# ---------------------------------------------------------------------


class LFlowPreprocessor:
    """
    Main preprocessing object used by training and inference transforms.
    """

    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        mel_config: Optional[MelConfig] = None,
        text_config: Optional[TextConfig] = None,
        mask_config: Optional[MaskConfig] = None,
    ) -> None:
        self.audio_config = audio_config or AudioConfig()
        self.mel_config = mel_config or MelConfig()
        self.text_config = text_config or TextConfig()
        self.mask_config = mask_config or MaskConfig()

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=self.audio_config.sample_rate,
            mel_config=self.mel_config,
        )

        self.tokenizer = CharacterTokenizer(
            lowercase=self.text_config.lowercase,
            filler_token=self.text_config.filler_token,
        )

    def process_audio(
        self,
        audio_path: PathLike,
        random_crop: bool = False,
    ) -> Dict[str, Any]:
        """
        Load one audio file and convert it into waveform and mel tensors.
        """

        waveform, sample_rate = load_audio(
            audio_path,
            target_sample_rate=self.audio_config.sample_rate,
            mono=True,
        )

        if self.audio_config.trim_silence:
            waveform = trim_silence(
                waveform,
                top_db=self.audio_config.silence_top_db,
            )

        waveform = crop_or_pad_waveform(
            waveform,
            sample_rate=sample_rate,
            max_audio_seconds=self.audio_config.max_audio_seconds,
            min_audio_seconds=self.audio_config.min_audio_seconds,
            random_crop=random_crop,
        )

        if self.audio_config.normalize_waveform:
            waveform = normalize_waveform(waveform)

        mel = self.mel_extractor(waveform)

        return {
            "waveform": waveform,
            "mel": mel,
            "sample_rate": sample_rate,
            "num_samples": waveform.shape[-1],
            "num_frames": mel.shape[-1],
        }

    def process_text(
        self,
        transcript: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert transcript into character token IDs.
        """

        token_ids = self.tokenizer.encode(transcript)

        if target_length is not None and self.text_config.pad_to_mel_length:
            token_tensor = self.tokenizer.pad_to_length(
                token_ids,
                target_length=target_length,
                pad_with_filler=True,
            )
        else:
            token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "text_ids": token_tensor,
            "text_length": min(len(token_ids), len(token_tensor)),
        }

    def create_training_mask(self, num_frames: int, n_mels: int) -> torch.Tensor:
        """
        Create a mel-shaped mask for training.
        """

        strategy = self.mask_config.strategy

        if strategy == "random_span":
            time_mask = create_random_span_mask(
                num_frames=num_frames,
                min_mask_ratio=self.mask_config.min_mask_ratio,
                max_mask_ratio=self.mask_config.max_mask_ratio,
            )
        elif strategy == "random":
            time_mask = create_random_mask(
                num_frames=num_frames,
                min_mask_ratio=self.mask_config.min_mask_ratio,
                max_mask_ratio=self.mask_config.max_mask_ratio,
            )
        elif strategy == "full":
            time_mask = create_full_mask(num_frames)
        else:
            raise ValueError(f"Unknown mask strategy: {strategy}")

        return expand_time_mask(time_mask, n_mels=n_mels)

    def create_inference_mask(self, num_frames: int, n_mels: int) -> torch.Tensor:
        """
        Create a full mel mask for inference-time full generation.
        """

        time_mask = create_full_mask(num_frames)
        return expand_time_mask(time_mask, n_mels=n_mels)


# ---------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------


class LFlowTrainingTransform:
    """
    Transform for samples returned by LongitudinalSpeechDataset.

    Input item structure from data.py:
        {
            "subject_id": ...,
            "target": {...},
            "context": [{...}, ...],
            "reference": {...},
            "num_context_sessions": ...
        }

    Output item includes:
        target_mel
        observed_mel
        mask
        text_ids
        reference_mel
        context_mels
        context_times
        context_labels
        target_time
        target_label
        sigma
        noisy_mel
        target_vector_field
    """

    def __init__(
        self,
        preprocessor: LFlowPreprocessor,
        random_crop: bool = True,
        load_context_audio: bool = True,
    ) -> None:
        self.preprocessor = preprocessor
        self.random_crop = random_crop
        self.load_context_audio = load_context_audio

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        target = item["target"]
        reference = item["reference"]
        context = item["context"]

        target_audio = self.preprocessor.process_audio(
            target["audio_path"],
            random_crop=self.random_crop,
        )
        target_mel = target_audio["mel"]

        n_mels, num_frames = target_mel.shape

        mask = self.preprocessor.create_training_mask(
            num_frames=num_frames,
            n_mels=n_mels,
        )

        observed_mel = apply_mel_mask(target_mel, mask)

        text = self.preprocessor.process_text(
            target["transcript"],
            target_length=num_frames,
        )

        reference_audio = self.preprocessor.process_audio(
            reference["audio_path"],
            random_crop=False,
        )

        reference_mel = pad_or_crop_mel(
            reference_audio["mel"],
            target_frames=num_frames,
            random_crop=False,
        )

        context_times: List[float] = []
        context_labels: List[float] = []
        context_mels: List[torch.Tensor] = []

        for context_session in context:
            context_times.append(float(context_session["timestamp"]))
            context_labels.append(float(context_session["label"]))

            if self.load_context_audio:
                context_audio = self.preprocessor.process_audio(
                    context_session["audio_path"],
                    random_crop=False,
                )
                context_mel = pad_or_crop_mel(
                    context_audio["mel"],
                    target_frames=num_frames,
                    random_crop=False,
                )
                context_mels.append(context_mel)

        sigma = sample_flow_step(batch_size=1)[0]
        noisy_mel, target_vector_field = create_noisy_path(
            target_mel=target_mel,
            sigma=sigma,
        )

        output = {
            "subject_id": item["subject_id"],
            "target_audio_path": target["audio_path"],
            "reference_audio_path": reference["audio_path"],

            "target_mel": target_mel,
            "observed_mel": observed_mel,
            "mask": mask,

            "text_ids": text["text_ids"],
            "text_length": text["text_length"],

            "reference_mel": reference_mel,

            "context_times": torch.tensor(context_times, dtype=torch.float32),
            "context_labels": torch.tensor(context_labels, dtype=torch.float32),

            "target_time": torch.tensor(float(target["timestamp"]), dtype=torch.float32),
            "target_label": torch.tensor(float(target["label"]), dtype=torch.float32),

            "sigma": sigma,
            "noisy_mel": noisy_mel,
            "target_vector_field": target_vector_field,

            "num_context_sessions": len(context),
        }

        if self.load_context_audio:
            output["context_mels"] = context_mels

        return output


class LFlowInferenceTransform:
    """
    Transform for samples returned by LongitudinalInferenceDataset.

    This creates full-mask inputs for generating complete synthetic samples at
    intermediate query timestamps.
    """

    def __init__(
        self,
        preprocessor: LFlowPreprocessor,
        target_num_frames: Optional[int] = None,
        load_context_audio: bool = True,
    ) -> None:
        self.preprocessor = preprocessor
        self.target_num_frames = target_num_frames
        self.load_context_audio = load_context_audio

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        reference = item["reference"]
        context = item["context"]

        reference_audio = self.preprocessor.process_audio(
            reference["audio_path"],
            random_crop=False,
        )

        reference_mel = reference_audio["mel"]

        if self.target_num_frames is not None:
            reference_mel = pad_or_crop_mel(
                reference_mel,
                target_frames=self.target_num_frames,
                random_crop=False,
            )

        n_mels, num_frames = reference_mel.shape

        # During inference, the entire target is masked and generated.
        mask = self.preprocessor.create_inference_mask(
            num_frames=num_frames,
            n_mels=n_mels,
        )

        observed_mel = torch.zeros_like(reference_mel)

        # For fixed-task datasets, the transcript can usually be taken from the
        # neighboring/reference session. If a custom query transcript is later
        # added to the item, use it instead.
        transcript = item.get("transcript", reference.get("transcript", ""))

        text = self.preprocessor.process_text(
            transcript,
            target_length=num_frames,
        )

        context_times: List[float] = []
        context_labels: List[float] = []
        context_mels: List[torch.Tensor] = []

        for context_session in context:
            context_times.append(float(context_session["timestamp"]))
            context_labels.append(float(context_session["label"]))

            if self.load_context_audio:
                context_audio = self.preprocessor.process_audio(
                    context_session["audio_path"],
                    random_crop=False,
                )
                context_mel = pad_or_crop_mel(
                    context_audio["mel"],
                    target_frames=num_frames,
                    random_crop=False,
                )
                context_mels.append(context_mel)

        output = {
            "subject_id": item["subject_id"],
            "query_timestamp": torch.tensor(
                float(item["query_timestamp"]),
                dtype=torch.float32,
            ),
            "soft_label": torch.tensor(
                float(item["soft_label"]),
                dtype=torch.float32,
            ),

            "previous_session": item["previous_session"],
            "next_session": item["next_session"],

            "reference_audio_path": reference["audio_path"],
            "reference_mel": reference_mel,

            "observed_mel": observed_mel,
            "mask": mask,

            "text_ids": text["text_ids"],
            "text_length": text["text_length"],

            "context_times": torch.tensor(context_times, dtype=torch.float32),
            "context_labels": torch.tensor(context_labels, dtype=torch.float32),

            "num_context_sessions": len(context),
        }

        if self.load_context_audio:
            output["context_mels"] = context_mels

        return output


# ---------------------------------------------------------------------
# Collate functions for tensor batching
# ---------------------------------------------------------------------


def pad_1d_sequence(
    sequences: Sequence[torch.Tensor],
    padding_value: int = 0,
) -> torch.Tensor:
    """
    Pad 1D tensors to the same length.

    Returns shape [batch, max_len].
    """

    if len(sequences) == 0:
        raise ValueError("sequences must not be empty.")

    max_len = max(seq.shape[0] for seq in sequences)
    batch = torch.full(
        (len(sequences), max_len),
        fill_value=padding_value,
        dtype=sequences[0].dtype,
    )

    for i, seq in enumerate(sequences):
        batch[i, : seq.shape[0]] = seq

    return batch


def pad_mel_sequence(
    mels: Sequence[torch.Tensor],
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad mel tensors to same number of frames.

    Input mel shape:
        [n_mels, frames]

    Output shape:
        [batch, n_mels, max_frames]
    """

    if len(mels) == 0:
        raise ValueError("mels must not be empty.")

    n_mels = mels[0].shape[0]
    max_frames = max(mel.shape[-1] for mel in mels)

    batch = torch.full(
        (len(mels), n_mels, max_frames),
        fill_value=padding_value,
        dtype=mels[0].dtype,
    )

    for i, mel in enumerate(mels):
        if mel.shape[0] != n_mels:
            raise ValueError("All mel tensors must have the same n_mels.")
        frames = mel.shape[-1]
        batch[i, :, :frames] = mel

    return batch


def pad_context_mels(
    context_mels_batch: Sequence[List[torch.Tensor]],
    padding_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length context mel lists.

    Input:
        batch of context_mels, each item is a list of [n_mels, frames].

    Output:
        context_mels:
            [batch, max_context, n_mels, max_frames]
        context_mask:
            [batch, max_context], where 1 means valid context session.
    """

    batch_size = len(context_mels_batch)
    max_context = max(len(x) for x in context_mels_batch)

    if max_context == 0:
        raise ValueError("Each item must have at least one context session.")

    first_mel = None
    for context_mels in context_mels_batch:
        if len(context_mels) > 0:
            first_mel = context_mels[0]
            break

    if first_mel is None:
        raise ValueError("No context mel tensors found.")

    n_mels = first_mel.shape[0]
    max_frames = max(
        mel.shape[-1]
        for context_mels in context_mels_batch
        for mel in context_mels
    )

    output = torch.full(
        (batch_size, max_context, n_mels, max_frames),
        fill_value=padding_value,
        dtype=first_mel.dtype,
    )
    context_mask = torch.zeros(batch_size, max_context, dtype=torch.float32)

    for i, context_mels in enumerate(context_mels_batch):
        for j, mel in enumerate(context_mels):
            frames = mel.shape[-1]
            output[i, j, :, :frames] = mel
            context_mask[i, j] = 1.0

    return output, context_mask


def pad_context_values(
    values_batch: Sequence[torch.Tensor],
    padding_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length context scalar values, such as times or labels.

    Returns
    -------
    values:
        [batch, max_context]
    mask:
        [batch, max_context], where 1 means valid.
    """

    batch_size = len(values_batch)
    max_len = max(v.shape[0] for v in values_batch)

    values = torch.full(
        (batch_size, max_len),
        fill_value=padding_value,
        dtype=torch.float32,
    )
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, v in enumerate(values_batch):
        length = v.shape[0]
        values[i, :length] = v
        mask[i, :length] = 1.0

    return values, mask


def lflow_training_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function after LFlowTrainingTransform.

    This pads variable-length mel/text/context tensors into batches.
    """

    target_mel = pad_mel_sequence([item["target_mel"] for item in batch])
    observed_mel = pad_mel_sequence([item["observed_mel"] for item in batch])
    mask = pad_mel_sequence([item["mask"] for item in batch])
    noisy_mel = pad_mel_sequence([item["noisy_mel"] for item in batch])
    target_vector_field = pad_mel_sequence(
        [item["target_vector_field"] for item in batch]
    )

    reference_mel = pad_mel_sequence([item["reference_mel"] for item in batch])

    text_ids = pad_1d_sequence(
        [item["text_ids"] for item in batch],
        padding_value=0,
    )

    context_times, context_mask = pad_context_values(
        [item["context_times"] for item in batch]
    )
    context_labels, _ = pad_context_values(
        [item["context_labels"] for item in batch]
    )

    output = {
        "subject_id": [item["subject_id"] for item in batch],
        "target_audio_path": [item["target_audio_path"] for item in batch],
        "reference_audio_path": [item["reference_audio_path"] for item in batch],

        "target_mel": target_mel,
        "observed_mel": observed_mel,
        "mask": mask,
        "noisy_mel": noisy_mel,
        "target_vector_field": target_vector_field,

        "text_ids": text_ids,
        "text_length": torch.tensor(
            [item["text_length"] for item in batch],
            dtype=torch.long,
        ),

        "reference_mel": reference_mel,

        "context_times": context_times,
        "context_labels": context_labels,
        "context_mask": context_mask,

        "target_time": torch.stack([item["target_time"] for item in batch]),
        "target_label": torch.stack([item["target_label"] for item in batch]),

        "sigma": torch.stack([item["sigma"] for item in batch]),
        "num_context_sessions": torch.tensor(
            [item["num_context_sessions"] for item in batch],
            dtype=torch.long,
        ),
    }

    if "context_mels" in batch[0]:
        context_mels, context_mel_mask = pad_context_mels(
            [item["context_mels"] for item in batch]
        )
        output["context_mels"] = context_mels
        output["context_mel_mask"] = context_mel_mask

    return output


def lflow_inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function after LFlowInferenceTransform.
    """

    observed_mel = pad_mel_sequence([item["observed_mel"] for item in batch])
    mask = pad_mel_sequence([item["mask"] for item in batch])
    reference_mel = pad_mel_sequence([item["reference_mel"] for item in batch])

    text_ids = pad_1d_sequence(
        [item["text_ids"] for item in batch],
        padding_value=0,
    )

    context_times, context_mask = pad_context_values(
        [item["context_times"] for item in batch]
    )
    context_labels, _ = pad_context_values(
        [item["context_labels"] for item in batch]
    )

    output = {
        "subject_id": [item["subject_id"] for item in batch],

        "query_timestamp": torch.stack(
            [item["query_timestamp"] for item in batch]
        ),
        "soft_label": torch.stack(
            [item["soft_label"] for item in batch]
        ),

        "previous_session": [item["previous_session"] for item in batch],
        "next_session": [item["next_session"] for item in batch],

        "reference_audio_path": [item["reference_audio_path"] for item in batch],
        "reference_mel": reference_mel,

        "observed_mel": observed_mel,
        "mask": mask,

        "text_ids": text_ids,
        "text_length": torch.tensor(
            [item["text_length"] for item in batch],
            dtype=torch.long,
        ),

        "context_times": context_times,
        "context_labels": context_labels,
        "context_mask": context_mask,

        "num_context_sessions": torch.tensor(
            [item["num_context_sessions"] for item in batch],
            dtype=torch.long,
        ),
    }

    if "context_mels" in batch[0]:
        context_mels, context_mel_mask = pad_context_mels(
            [item["context_mels"] for item in batch]
        )
        output["context_mels"] = context_mels
        output["context_mel_mask"] = context_mel_mask

    return output


# ---------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------


def build_preprocessor_from_config(config: Dict[str, Any]) -> LFlowPreprocessor:
    """
    Create LFlowPreprocessor from a loaded YAML config.
    """

    preprocessing_cfg = config.get("preprocessing", {})

    mel_cfg = preprocessing_cfg.get("mel", {})
    text_cfg = preprocessing_cfg.get("text", {})

    flow_cfg = config.get("flow_matching", {})
    mask_cfg = flow_cfg.get("mask", {})

    audio_config = AudioConfig(
        sample_rate=int(preprocessing_cfg.get("sample_rate", 16000)),
        max_audio_seconds=preprocessing_cfg.get("max_audio_seconds", 30.0),
        min_audio_seconds=float(preprocessing_cfg.get("min_audio_seconds", 1.0)),
        trim_silence=bool(preprocessing_cfg.get("trim_silence", True)),
    )

    mel_config = MelConfig(
        n_fft=int(mel_cfg.get("n_fft", 1024)),
        hop_length=int(mel_cfg.get("hop_length", 256)),
        win_length=int(mel_cfg.get("win_length", 1024)),
        n_mels=int(mel_cfg.get("n_mels", 100)),
        f_min=float(mel_cfg.get("f_min", 0.0)),
        f_max=mel_cfg.get("f_max", 8000),
        log_mel=bool(mel_cfg.get("log_mel", True)),
    )

    text_config = TextConfig(
        lowercase=bool(text_cfg.get("lowercase", True)),
        use_char_level=bool(text_cfg.get("use_char_level", True)),
        filler_token=str(text_cfg.get("filler_token", "<F>")),
        pad_to_mel_length=bool(text_cfg.get("pad_to_mel_length", True)),
    )

    mask_config = MaskConfig(
        strategy=str(mask_cfg.get("strategy", "random_span")),
        min_mask_ratio=float(mask_cfg.get("min_mask_ratio", 0.3)),
        max_mask_ratio=float(mask_cfg.get("max_mask_ratio", 0.8)),
    )

    return LFlowPreprocessor(
        audio_config=audio_config,
        mel_config=mel_config,
        text_config=text_config,
        mask_config=mask_config,
    )