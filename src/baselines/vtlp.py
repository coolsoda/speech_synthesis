"""

Vocal Tract Length Perturbation baseline for L-Flow.

This module generates augmented speech samples by applying a lightweight
frequency-warping approximation of vocal tract length perturbation (VTLP) to
real reference recordings.

The generated metadata follows the same format as L-Flow synthetic metadata:

    method,subject_id,timestamp,label,transcript,audio_path,
    reference_audio_path,previous_timestamp,next_timestamp,
    previous_label,next_label

"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
from torch.utils.data import DataLoader

from lflow.data import LongitudinalInferenceDataset
from lflow.utils import (
    ensure_dir,
    save_csv_rows,
    save_json,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------


def load_waveform(
    audio_path: str,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load audio and resample to target sample rate.

    Returns
    -------
    waveform:
        Tensor with shape [1, T].
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=sample_rate,
        )

    max_abs = waveform.abs().max()
    if max_abs > 0:
        waveform = waveform / max_abs

    return waveform


def save_waveform(
    waveform: torch.Tensor,
    output_path: Path,
    sample_rate: int = 16000,
) -> None:
    """
    Save waveform to disk.
    """

    ensure_dir(output_path.parent)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.detach().cpu().clamp(-1.0, 1.0)

    torchaudio.save(
        str(output_path),
        waveform,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------
# VTLP implementation
# ---------------------------------------------------------------------


def apply_vtlp(
    waveform: torch.Tensor,
    warp_factor: float,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """
    Apply a lightweight VTLP-style spectral warping.

    This implementation performs frequency-axis interpolation on the STFT
    magnitude and reconstructs audio using the original phase.

    Parameters
    ----------
    waveform:
        Tensor with shape [1, T].
    warp_factor:
        Frequency-warping factor. Values around 0.9 or 1.1 are typical.
    sample_rate:
        Audio sample rate.
    n_fft:
        FFT size.
    hop_length:
        Hop size.
    win_length:
        Window length.

    Returns
    -------
    perturbed:
        VTLP-augmented waveform with shape [1, T].
    """

    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [channels, samples].")

    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if warp_factor <= 0:
        raise ValueError("warp_factor must be positive.")

    device = waveform.device
    dtype = waveform.dtype

    window = torch.hann_window(
        win_length,
        device=device,
        dtype=dtype,
    )

    # STFT shape: [freq_bins, frames]
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    magnitude = stft.abs()
    phase = torch.angle(stft)

    warped_magnitude = warp_frequency_axis(
        magnitude=magnitude,
        warp_factor=warp_factor,
    )

    warped_stft = torch.polar(warped_magnitude, phase)

    reconstructed = torch.istft(
        warped_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=waveform.shape[-1],
    )

    reconstructed = reconstructed.unsqueeze(0)

    max_abs = reconstructed.abs().max()
    if max_abs > 0:
        reconstructed = reconstructed / max_abs

    return reconstructed


def warp_frequency_axis(
    magnitude: torch.Tensor,
    warp_factor: float,
) -> torch.Tensor:
    """
    Warp the frequency axis of a magnitude spectrogram.

    Input:
        magnitude: [freq_bins, frames]

    Output:
        warped_magnitude: [freq_bins, frames]

    The operation maps target frequency bins back to source frequency bins:

        source_bin = target_bin / warp_factor

    If source_bin is outside the valid range, values are clamped.
    """

    if magnitude.ndim != 2:
        raise ValueError(
            f"Expected magnitude shape [freq_bins, frames], got {tuple(magnitude.shape)}."
        )

    freq_bins, frames = magnitude.shape
    device = magnitude.device
    dtype = magnitude.dtype

    target_bins = torch.arange(freq_bins, device=device, dtype=dtype)
    source_bins = target_bins / float(warp_factor)
    source_bins = source_bins.clamp(0, freq_bins - 1)

    lower = torch.floor(source_bins).long()
    upper = torch.ceil(source_bins).long()

    weight = (source_bins - lower.to(dtype)).view(freq_bins, 1)

    lower_values = magnitude[lower, :]
    upper_values = magnitude[upper, :]

    warped = (1.0 - weight) * lower_values + weight * upper_values

    return warped


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------


def choose_warp_factor(warp_factors: List[float]) -> float:
    """
    Select a VTLP warp factor from the configured list.
    """

    if len(warp_factors) == 0:
        raise ValueError("warp_factors must not be empty.")

    return float(random.choice(warp_factors))


def make_output_filename(
    subject_id: str,
    query_timestamp: float,
    index: int,
    warp_factor: float,
) -> str:
    """
    Make a stable output filename.
    """

    safe_subject = str(subject_id).replace("/", "_").replace("\\", "_")
    safe_time = f"{query_timestamp:.4f}".replace(".", "p")
    safe_factor = f"{warp_factor:.2f}".replace(".", "p")

    return f"{safe_subject}_t{safe_time}_vtlp{safe_factor}_{index:06d}.wav"


def build_metadata_row(
    method: str,
    subject_id: str,
    query_timestamp: float,
    soft_label: float,
    transcript: str,
    audio_path: str,
    reference_audio_path: str,
    previous_session: Dict[str, Any],
    next_session: Dict[str, Any],
    warp_factor: float,
) -> Dict[str, Any]:
    """
    Build one synthetic metadata row.
    """

    return {
        "method": method,
        "subject_id": subject_id,
        "timestamp": query_timestamp,
        "label": soft_label,
        "transcript": transcript,
        "audio_path": audio_path,
        "reference_audio_path": reference_audio_path,
        "previous_timestamp": previous_session.get("timestamp"),
        "next_timestamp": next_session.get("timestamp"),
        "previous_label": previous_session.get("label"),
        "next_label": next_session.get("label"),
        "warp_factor": warp_factor,
    }


# ---------------------------------------------------------------------
# Dataset/dataloader
# ---------------------------------------------------------------------


def build_baseline_query_loader(
    global_config: Dict[str, Any],
) -> DataLoader:
    """
    Build interpolation query dataloader for baseline generation.

    This uses the same LongitudinalInferenceDataset as L-Flow so that the
    number and placement of generated samples match the main method.
    """

    dataset_cfg = global_config.get("dataset", {})
    generation_cfg = global_config.get("generation", {})

    split = dataset_cfg.get("generate_for_split", "train")

    dataset = LongitudinalInferenceDataset(
        metadata_path=dataset_cfg["metadata_path"],
        split_path=dataset_cfg.get("split_path"),
        split=split,
        root_dir=dataset_cfg.get("root_dir"),
        subject_column=dataset_cfg.get("subject_column", "subject_id"),
        audio_column=dataset_cfg.get("audio_column", "audio_path"),
        transcript_column=dataset_cfg.get("transcript_column", "transcript"),
        time_column=dataset_cfg.get("time_column", "timestamp"),
        label_column=dataset_cfg.get("label_column", "label"),
        min_sessions=dataset_cfg.get("min_sessions", 2),
        temporal_resolution=generation_cfg.get("temporal_resolution_days", 30),
        max_samples_per_interval=generation_cfg.get("max_samples_per_interval", 2),
        time_unit=generation_cfg.get("time_unit", "days"),
        transform=None,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def run_vtlp(
    global_config: Dict[str, Any],
    method_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run VTLP baseline.

    Parameters
    ----------
    global_config:
        Full baselines.yaml config.
    method_config:
        Config section for baselines.signal_level.vtlp.

    Returns
    -------
    dict
        Summary of generated files.
    """

    experiment_cfg = global_config.get("experiment", {})
    generation_cfg = global_config.get("generation", {})
    output_cfg = global_config.get("output", {})

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    sample_rate = int(generation_cfg.get("sample_rate", 16000))
    audio_format = generation_cfg.get("audio_format", "wav")

    if audio_format.lower() != "wav":
        raise ValueError("vtlp.py currently supports only wav output.")

    output_dir = Path(
        method_config.get(
            "output_dir",
            Path(experiment_cfg.get("output_dir", "outputs/generated_audio/baselines"))
            / "vtlp",
        )
    )

    audio_dir = output_dir / "wav"
    metadata_path = output_dir / output_cfg.get(
        "metadata_filename",
        "synthetic_metadata.csv",
    )

    ensure_dir(audio_dir)

    logger = setup_logger(
        name="baselines.vtlp",
        log_file=output_dir / "vtlp.log",
    )

    logger.info("Running VTLP baseline")
    logger.info(f"Output directory: {output_dir}")

    warp_factors = method_config.get("warp_factors", [0.9, 1.1])
    warp_factors = [float(x) for x in warp_factors]

    n_fft = int(method_config.get("n_fft", 1024))
    hop_length = int(method_config.get("hop_length", 256))
    win_length = int(method_config.get("win_length", 1024))

    loader = build_baseline_query_loader(global_config)

    rows: List[Dict[str, Any]] = []
    num_generated = 0
    num_failed = 0

    for batch_idx, batch in enumerate(loader):
        item = batch[0]

        subject_id = item["subject_id"]
        query_timestamp = float(item["query_timestamp"])
        soft_label = float(item["soft_label"])

        previous_session = item["previous_session"]
        next_session = item["next_session"]
        reference_session = item["reference"]

        reference_audio_path = reference_session["audio_path"]

        transcript = reference_session.get("transcript", "")
        if not transcript:
            transcript = previous_session.get("transcript", "")
        if not transcript:
            transcript = next_session.get("transcript", "")

        warp_factor = choose_warp_factor(warp_factors)

        try:
            waveform = load_waveform(
                reference_audio_path,
                sample_rate=sample_rate,
            )

            perturbed = apply_vtlp(
                waveform=waveform,
                warp_factor=warp_factor,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

            filename = make_output_filename(
                subject_id=subject_id,
                query_timestamp=query_timestamp,
                index=num_generated,
                warp_factor=warp_factor,
            )

            output_path = audio_dir / filename

            save_waveform(
                perturbed,
                output_path=output_path,
                sample_rate=sample_rate,
            )

            row = build_metadata_row(
                method="vtlp",
                subject_id=subject_id,
                query_timestamp=query_timestamp,
                soft_label=soft_label,
                transcript=transcript,
                audio_path=str(output_path),
                reference_audio_path=reference_audio_path,
                previous_session=previous_session,
                next_session=next_session,
                warp_factor=warp_factor,
            )

            rows.append(row)
            num_generated += 1

        except Exception as exc:
            num_failed += 1
            logger.warning(
                f"Failed to generate VTLP sample for subject={subject_id}, "
                f"time={query_timestamp}: {exc}"
            )

        if (batch_idx + 1) % int(generation_cfg.get("log_interval", 100)) == 0:
            logger.info(
                f"Processed {batch_idx + 1}/{len(loader)} queries | "
                f"generated={num_generated}, failed={num_failed}"
            )

    fieldnames = [
        "method",
        "subject_id",
        "timestamp",
        "label",
        "transcript",
        "audio_path",
        "reference_audio_path",
        "previous_timestamp",
        "next_timestamp",
        "previous_label",
        "next_label",
        "warp_factor",
    ]

    save_csv_rows(
        rows=rows,
        path=metadata_path,
        fieldnames=fieldnames,
    )

    summary = {
        "method": "vtlp",
        "num_queries": len(loader.dataset),
        "num_generated": num_generated,
        "num_failed": num_failed,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "warp_factors": warp_factors,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
    }

    save_json(summary, output_dir / "summary.json")

    logger.info(
        f"Finished VTLP baseline | generated={num_generated}, failed={num_failed}"
    )
    logger.info(f"Saved metadata to {metadata_path}")

    return summary