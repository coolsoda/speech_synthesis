"""

1. Loading a trained L-Flow checkpoint.
2. Building interpolation queries between neighboring real sessions.
3. Generating synthetic mel-spectrograms.
4. Optionally vocoding generated mels into waveform audio.
5. Saving generated samples and synthetic metadata.

The main entry point is:

    generate_synthetic_dataset(config)

This file is intended to be imported by:
    src/scripts/generate_synthetic.py
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader

from .data import LongitudinalInferenceDataset
from .generator import build_lflow_model_from_config
from .preprocessing import (
    LFlowInferenceTransform,
    build_preprocessor_from_config,
    lflow_inference_collate_fn,
)
from .utils import (
    ensure_dir,
    get_device_from_config,
    load_checkpoint,
    load_yaml,
    move_to_device,
    save_csv_rows,
    save_json,
    save_yaml,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------
# Vocoder wrapper
# ---------------------------------------------------------------------


class VocoderWrapper:
    """
    Lightweight vocoder wrapper.

    This class is intentionally conservative. In a real reproduction setting,
    users may plug in their own HiFi-GAN or F5-TTS-compatible vocoder.

    Supported modes:
        1. If a TorchScript vocoder checkpoint is provided, load it with torch.jit.load.
        2. If no valid vocoder is available, save generated mel tensors only.

    Expected vocoder input:
        mel: [B, F, N]

    Expected vocoder output:
        waveform: [B, T] or [B, 1, T]
    """

    def __init__(
        self,
        checkpoint_path: Optional[str],
        device: torch.device,
        sample_rate: int = 16000,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.sample_rate = sample_rate
        self.model: Optional[torch.nn.Module] = None

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)

            if checkpoint.exists():
                try:
                    self.model = torch.jit.load(str(checkpoint), map_location=device)
                    self.model.eval()
                    self.model.to(device)
                except Exception:
                    self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def mel_to_waveform(self, mel: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert generated mel to waveform.

        Returns None if no compatible vocoder is available.
        """

        if self.model is None:
            return None

        mel = mel.to(self.device)

        waveform = self.model(mel)

        if isinstance(waveform, (tuple, list)):
            waveform = waveform[0]

        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.clamp(-1.0, 1.0)

        return waveform


# ---------------------------------------------------------------------
# Checkpoint/model loading
# ---------------------------------------------------------------------


def load_lflow_for_inference(
    config: Dict[str, Any],
    device: torch.device,
    vocab_size: int,
    checkpoint_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Build L-Flow model and load trained checkpoint.
    """

    if checkpoint_path is None:
        checkpoint_cfg = config.get("checkpoint", {})
        checkpoint_path = checkpoint_cfg.get("lflow_path")

    if checkpoint_path is None:
        raise ValueError(
            "No L-Flow checkpoint path provided. "
            "Set checkpoint.lflow_path in inference.yaml."
        )

    model = build_lflow_model_from_config(
        config=config,
        vocab_size=vocab_size,
    )

    model = model.to(device)

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=device,
        strict=True,
    )

    model.eval()
    return model


# ---------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------


def build_inference_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, Any]:
    """
    Build dataloader for synthetic generation.
    """

    dataset_cfg = config.get("dataset", {})
    generation_cfg = config.get("generation", {})

    preprocessor = build_preprocessor_from_config(config)

    target_num_frames = generation_cfg.get("target_num_frames")
    if target_num_frames is not None:
        target_num_frames = int(target_num_frames)

    transform = LFlowInferenceTransform(
        preprocessor=preprocessor,
        target_num_frames=target_num_frames,
        load_context_audio=True,
    )

    split = dataset_cfg.get(
        "generate_for_split",
        dataset_cfg.get("train_split", "train"),
    )

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
        transform=transform,
    )

    batch_size = int(generation_cfg.get("batch_size", 4))
    num_workers = int(generation_cfg.get("num_workers", 2))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lflow_inference_collate_fn,
    )

    return dataloader, preprocessor


# ---------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------


def save_generated_mel(
    mel: torch.Tensor,
    output_path: Path,
) -> None:
    """
    Save one generated mel tensor as .pt.
    """

    ensure_dir(output_path.parent)
    torch.save(mel.detach().cpu(), output_path)


def save_generated_audio(
    waveform: torch.Tensor,
    output_path: Path,
    sample_rate: int,
) -> None:
    """
    Save one generated waveform as .wav.
    """

    ensure_dir(output_path.parent)

    waveform = waveform.detach().cpu()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim == 2 and waveform.shape[0] != 1:
        waveform = waveform[:1]

    torchaudio.save(
        str(output_path),
        waveform,
        sample_rate=sample_rate,
    )


def make_generated_filename(
    subject_id: str,
    query_timestamp: float,
    index: int,
    extension: str,
) -> str:
    """
    Create a stable generated filename.

    Timestamps may be large floating point values, so we sanitize them.
    """

    safe_subject = str(subject_id).replace("/", "_").replace("\\", "_")
    safe_time = f"{query_timestamp:.4f}".replace(".", "p")
    return f"{safe_subject}_t{safe_time}_{index:06d}.{extension}"


def batch_to_metadata_rows(
    batch: Dict[str, Any],
    generated_audio_paths: List[Optional[str]],
    generated_mel_paths: List[Optional[str]],
    method_name: str = "lflow",
) -> List[Dict[str, Any]]:
    """
    Convert an inference batch and saved paths into metadata rows.
    """

    rows: List[Dict[str, Any]] = []

    batch_size = len(batch["subject_id"])

    query_timestamps = batch["query_timestamp"].detach().cpu().tolist()
    soft_labels = batch["soft_label"].detach().cpu().tolist()

    for i in range(batch_size):
        previous_session = batch["previous_session"][i]
        next_session = batch["next_session"][i]

        transcript = previous_session.get("transcript", "")
        if not transcript:
            transcript = next_session.get("transcript", "")

        row = {
            "method": method_name,
            "subject_id": batch["subject_id"][i],
            "timestamp": query_timestamps[i],
            "label": soft_labels[i],
            "transcript": transcript,
            "audio_path": generated_audio_paths[i],
            "mel_path": generated_mel_paths[i],
            "reference_audio_path": batch["reference_audio_path"][i],
            "previous_timestamp": previous_session.get("timestamp"),
            "next_timestamp": next_session.get("timestamp"),
            "previous_label": previous_session.get("label"),
            "next_label": next_session.get("label"),
        }

        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------


@torch.no_grad()
def generate_synthetic_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate synthetic speech samples using a trained L-Flow model.

    Parameters
    ----------
    config:
        Loaded inference YAML config.

    Returns
    -------
    dict
        Generation summary.
    """

    experiment_cfg = config.get("experiment", {})
    generation_cfg = config.get("generation", {})
    output_cfg = config.get("output", {})
    checkpoint_cfg = config.get("checkpoint", {})
    preprocessing_cfg = config.get("preprocessing", {})

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(
        experiment_cfg.get(
            "output_dir",
            "outputs/generated_audio/lflow",
        )
    )

    mel_dir = output_dir / "mels"
    audio_dir = output_dir / "wav"
    metadata_path = output_dir / output_cfg.get(
        "metadata_filename",
        "synthetic_metadata.csv",
    )

    ensure_dir(output_dir)

    if output_cfg.get("save_mel", True):
        ensure_dir(mel_dir)

    if output_cfg.get("save_audio", True):
        ensure_dir(audio_dir)

    logger = setup_logger(
        name="lflow.inference",
        log_file=output_dir / "inference.log",
    )

    logger.info("Starting L-Flow synthetic generation")
    logger.info(f"Output directory: {output_dir}")

    save_yaml(config, output_dir / "inference_config.yaml")

    device = get_device_from_config(config)
    logger.info(f"Using device: {device}")

    dataloader, preprocessor = build_inference_dataloader(config)

    logger.info(f"Number of generation batches: {len(dataloader)}")
    logger.info(f"Number of generation queries: {len(dataloader.dataset)}")

    vocab_size = preprocessor.tokenizer.vocab_size

    model = load_lflow_for_inference(
        config=config,
        device=device,
        vocab_size=vocab_size,
        checkpoint_path=checkpoint_cfg.get("lflow_path"),
    )

    vocoder_path = checkpoint_cfg.get("vocoder_path")
    sample_rate = int(preprocessing_cfg.get("sample_rate", 16000))

    vocoder = VocoderWrapper(
        checkpoint_path=vocoder_path,
        device=device,
        sample_rate=sample_rate,
    )

    if output_cfg.get("save_audio", True) and not vocoder.available:
        logger.warning(
            "No compatible TorchScript vocoder was loaded. "
            "Generated mel tensors will be saved, but audio_path will be None."
        )

    solver_cfg = generation_cfg.get("ode_solver", {})
    solver = solver_cfg.get("name", "euler")
    num_steps = int(solver_cfg.get("num_steps", 32))

    save_audio = bool(output_cfg.get("save_audio", True))
    save_mel = bool(output_cfg.get("save_mel", True))

    all_rows: List[Dict[str, Any]] = []
    total_generated = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = move_to_device(batch, device)

        generated_mel = model.generate_mel(
            batch=batch,
            num_steps=num_steps,
            solver=solver,
        )

        generated_waveform = None
        if save_audio and vocoder.available:
            generated_waveform = vocoder.mel_to_waveform(generated_mel)

        batch_size = generated_mel.shape[0]

        generated_audio_paths: List[Optional[str]] = []
        generated_mel_paths: List[Optional[str]] = []

        for i in range(batch_size):
            subject_id = batch["subject_id"][i]
            query_timestamp = float(batch["query_timestamp"][i].detach().cpu())
            global_index = total_generated + i

            mel_path_str: Optional[str] = None
            audio_path_str: Optional[str] = None

            if save_mel:
                mel_filename = make_generated_filename(
                    subject_id=subject_id,
                    query_timestamp=query_timestamp,
                    index=global_index,
                    extension="pt",
                )
                mel_path = mel_dir / mel_filename
                save_generated_mel(generated_mel[i], mel_path)
                mel_path_str = str(mel_path)

            if save_audio and generated_waveform is not None:
                audio_filename = make_generated_filename(
                    subject_id=subject_id,
                    query_timestamp=query_timestamp,
                    index=global_index,
                    extension="wav",
                )
                audio_path = audio_dir / audio_filename

                save_generated_audio(
                    waveform=generated_waveform[i],
                    output_path=audio_path,
                    sample_rate=sample_rate,
                )
                audio_path_str = str(audio_path)

            generated_audio_paths.append(audio_path_str)
            generated_mel_paths.append(mel_path_str)

        rows = batch_to_metadata_rows(
            batch=batch,
            generated_audio_paths=generated_audio_paths,
            generated_mel_paths=generated_mel_paths,
            method_name="lflow",
        )

        all_rows.extend(rows)
        total_generated += batch_size

        if (batch_idx + 1) % int(generation_cfg.get("log_interval", 10)) == 0:
            logger.info(
                f"Generated batch {batch_idx + 1}/{len(dataloader)} | "
                f"Total samples: {total_generated}"
            )

    if output_cfg.get("save_metadata", True):
        fieldnames = [
            "method",
            "subject_id",
            "timestamp",
            "label",
            "transcript",
            "audio_path",
            "mel_path",
            "reference_audio_path",
            "previous_timestamp",
            "next_timestamp",
            "previous_label",
            "next_label",
        ]

        save_csv_rows(
            rows=all_rows,
            path=metadata_path,
            fieldnames=fieldnames,
        )

        logger.info(f"Saved synthetic metadata to {metadata_path}")

    summary = {
        "num_generated": total_generated,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "save_audio": save_audio,
        "save_mel": save_mel,
        "vocoder_available": vocoder.available,
        "solver": solver,
        "num_steps": num_steps,
    }

    save_json(summary, output_dir / "generation_summary.json")

    logger.info(f"Finished generation. Total samples: {total_generated}")

    return summary


# ---------------------------------------------------------------------
# Optional helper: copy generated metadata to a standard location
# ---------------------------------------------------------------------


def copy_synthetic_metadata(
    generated_metadata_path: str,
    destination_path: str,
) -> None:
    """
    Copy synthetic metadata to another location.

    Useful when linking inference output to downstream.yaml.
    """

    src = Path(generated_metadata_path)
    dst = Path(destination_path)

    if not src.exists():
        raise FileNotFoundError(f"Generated metadata not found: {src}")

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------


def generate_from_config_path(config_path: str) -> Dict[str, Any]:
    """
    Load inference YAML and run synthetic generation.
    """

    config = load_yaml(config_path)
    return generate_synthetic_dataset(config)