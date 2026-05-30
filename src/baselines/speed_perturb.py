"""

Speed perturbation baseline for L-Flow.

This module generates augmented speech samples by applying speed perturbation
to real reference recordings.
-----
Speed perturbation is a signal-level augmentation method. It does not model
longitudinal disease progression directly. For each intermediate query point,
we select a temporally nearby real recording as the reference audio and apply
a mild speed factor.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader

from lflow.data import LongitudinalInferenceDataset
from lflow.utils import (
    ensure_dir,
    get_device_from_config,
    save_csv_rows,
    save_json,
    set_seed,
    setup_logger,
)



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


def apply_speed_perturbation(
    waveform: torch.Tensor,
    speed_factor: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Apply speed perturbation by resampling.

    A speed_factor > 1.0 makes speech faster.
    A speed_factor < 1.0 makes speech slower.

    We implement this by:
        1. resampling from sample_rate to sample_rate * speed_factor;
        2. treating the resulting waveform as if it were still sample_rate.

    This changes duration and speaking rate while preserving file sample rate.
    """

    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive.")

    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [channels, samples].")

    intermediate_rate = max(1, int(round(sample_rate * speed_factor)))

    perturbed = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=intermediate_rate,
    )

    # Keep amplitude stable.
    max_abs = perturbed.abs().max()
    if max_abs > 0:
        perturbed = perturbed / max_abs

    return perturbed


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
# Metadata helpers
# ---------------------------------------------------------------------


def make_output_filename(
    subject_id: str,
    query_timestamp: float,
    index: int,
    speed_factor: float,
) -> str:
    """
    Make a stable output filename.
    """

    safe_subject = str(subject_id).replace("/", "_").replace("\\", "_")
    safe_time = f"{query_timestamp:.4f}".replace(".", "p")
    safe_factor = f"{speed_factor:.2f}".replace(".", "p")

    return f"{safe_subject}_t{safe_time}_speed{safe_factor}_{index:06d}.wav"


def choose_speed_factor(speed_factors: List[float]) -> float:
    """
    Select a speed factor from the configured list.
    """

    if len(speed_factors) == 0:
        raise ValueError("speed_factors must not be empty.")

    return float(random.choice(speed_factors))


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
    speed_factor: float,
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
        "speed_factor": speed_factor,
    }


# ---------------------------------------------------------------------
# Dataset/dataloader
# ---------------------------------------------------------------------


def build_baseline_query_loader(
    global_config: Dict[str, Any],
) -> DataLoader:
    """
    Build interpolation query dataloader for baseline generation.

    This uses the same LongitudinalInferenceDataset as L-Flow, but without
    loading mel tensors. Each item provides:
        - query timestamp;
        - soft label;
        - previous and next real sessions;
        - reference session.
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


def run_speed_perturbation(
    global_config: Dict[str, Any],
    method_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run speed perturbation baseline.

    Parameters
    ----------
    global_config:
        Full baselines.yaml config.
    method_config:
        Config section for baselines.signal_level.speed_perturbation.

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
        raise ValueError("speed_perturb.py currently supports only wav output.")

    output_dir = Path(
        method_config.get(
            "output_dir",
            Path(experiment_cfg.get("output_dir", "outputs/generated_audio/baselines"))
            / "speed_perturbation",
        )
    )

    audio_dir = output_dir / "wav"
    metadata_path = output_dir / output_cfg.get(
        "metadata_filename",
        "synthetic_metadata.csv",
    )

    ensure_dir(audio_dir)

    logger = setup_logger(
        name="baselines.speed_perturbation",
        log_file=output_dir / "speed_perturbation.log",
    )

    logger.info("Running speed perturbation baseline")
    logger.info(f"Output directory: {output_dir}")

    speed_factors = method_config.get("speed_factors", [0.9, 1.1])
    speed_factors = [float(x) for x in speed_factors]

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

        speed_factor = choose_speed_factor(speed_factors)

        try:
            waveform = load_waveform(
                reference_audio_path,
                sample_rate=sample_rate,
            )

            perturbed = apply_speed_perturbation(
                waveform=waveform,
                speed_factor=speed_factor,
                sample_rate=sample_rate,
            )

            filename = make_output_filename(
                subject_id=subject_id,
                query_timestamp=query_timestamp,
                index=num_generated,
                speed_factor=speed_factor,
            )

            output_path = audio_dir / filename

            save_waveform(
                perturbed,
                output_path=output_path,
                sample_rate=sample_rate,
            )

            row = build_metadata_row(
                method="speed_perturbation",
                subject_id=subject_id,
                query_timestamp=query_timestamp,
                soft_label=soft_label,
                transcript=transcript,
                audio_path=str(output_path),
                reference_audio_path=reference_audio_path,
                previous_session=previous_session,
                next_session=next_session,
                speed_factor=speed_factor,
            )

            rows.append(row)
            num_generated += 1

        except Exception as exc:
            num_failed += 1
            logger.warning(
                f"Failed to generate sample for subject={subject_id}, "
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
        "speed_factor",
    ]

    save_csv_rows(
        rows=rows,
        path=metadata_path,
        fieldnames=fieldnames,
    )

    summary = {
        "method": "speed_perturbation",
        "num_queries": len(loader.dataset),
        "num_generated": num_generated,
        "num_failed": num_failed,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "speed_factors": speed_factors,
    }

    save_json(summary, output_dir / "summary.json")

    logger.info(
        f"Finished speed perturbation baseline | "
        f"generated={num_generated}, failed={num_failed}"
    )
    logger.info(f"Saved metadata to {metadata_path}")

    return summary