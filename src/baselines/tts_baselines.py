"""

Wrapper interface for external TTS baseline systems.

This module provides a standardized interface for running external TTS or
pathology-oriented TTS baselines used in the L-Flow paper, including:

    - OmniVoice
    - Qwen3-TTS
    - VoxCPM2
    - XTTS-v2
    - TTDS
    - SCP-TTS
-------------------------------------
Each external baseline should read the manifest CSV and generate one audio file
per row. The manifest contains:

    sample_id
    subject_id
    timestamp
    label
    transcript
    reference_audio_path
    previous_audio_path
    next_audio_path
    output_audio_path

The external generator should save audio to output_audio_path.

The final metadata written by this wrapper follows:

    method
    subject_id
    timestamp
    label
    transcript
    audio_path
    reference_audio_path
    previous_timestamp
    next_timestamp
    previous_label
    next_label
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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
# Query dataset
# ---------------------------------------------------------------------


def build_baseline_query_loader(
    global_config: Dict[str, Any],
) -> DataLoader:
    """
    Build interpolation query dataloader for external TTS baselines.

    This uses the same LongitudinalInferenceDataset as L-Flow so that each
    baseline receives the same query timestamps and soft labels.
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
# Manifest creation
# ---------------------------------------------------------------------


def make_output_filename(
    method_name: str,
    subject_id: str,
    query_timestamp: float,
    index: int,
    extension: str = "wav",
) -> str:
    """
    Make a stable output filename for generated audio.
    """

    safe_method = str(method_name).replace("/", "_").replace("\\", "_")
    safe_subject = str(subject_id).replace("/", "_").replace("\\", "_")
    safe_time = f"{query_timestamp:.4f}".replace(".", "p")

    return f"{safe_method}_{safe_subject}_t{safe_time}_{index:06d}.{extension}"


def build_generation_manifest(
    method_name: str,
    global_config: Dict[str, Any],
    method_config: Dict[str, Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Build a manifest describing the samples an external TTS system should generate.

    Returns
    -------
    List[Dict[str, Any]]
        Manifest rows.
    """

    generation_cfg = global_config.get("generation", {})

    audio_format = generation_cfg.get("audio_format", "wav")
    audio_dir = output_dir / "wav"
    ensure_dir(audio_dir)

    loader = build_baseline_query_loader(global_config)

    rows: List[Dict[str, Any]] = []

    for index, batch in enumerate(loader):
        item = batch[0]

        subject_id = item["subject_id"]
        query_timestamp = float(item["query_timestamp"])
        soft_label = float(item["soft_label"])

        previous_session = item["previous_session"]
        next_session = item["next_session"]
        reference_session = item["reference"]

        transcript = reference_session.get("transcript", "")
        if not transcript:
            transcript = previous_session.get("transcript", "")
        if not transcript:
            transcript = next_session.get("transcript", "")

        output_filename = make_output_filename(
            method_name=method_name,
            subject_id=subject_id,
            query_timestamp=query_timestamp,
            index=index,
            extension=audio_format,
        )

        output_audio_path = audio_dir / output_filename

        row = {
            "sample_id": f"{method_name}_{index:06d}",
            "method": method_name,
            "subject_id": subject_id,
            "timestamp": query_timestamp,
            "label": soft_label,
            "transcript": transcript,
            "reference_audio_path": reference_session.get("audio_path"),
            "previous_audio_path": previous_session.get("audio_path"),
            "next_audio_path": next_session.get("audio_path"),
            "previous_timestamp": previous_session.get("timestamp"),
            "next_timestamp": next_session.get("timestamp"),
            "previous_label": previous_session.get("label"),
            "next_label": next_session.get("label"),
            "output_audio_path": str(output_audio_path),
        }

        # Optional model-specific conditioning fields.
        if method_config.get("use_severity_condition", False):
            row["severity_condition"] = soft_label

        if method_config.get("use_reference_speech", True):
            row["speaker_reference"] = reference_session.get("audio_path")

        if method_config.get("use_text_prompt", True):
            row["text_prompt"] = transcript

        rows.append(row)

    return rows


def save_manifest(
    rows: List[Dict[str, Any]],
    manifest_path: Path,
) -> None:
    """
    Save external generation manifest.
    """

    fieldnames = [
        "sample_id",
        "method",
        "subject_id",
        "timestamp",
        "label",
        "transcript",
        "reference_audio_path",
        "previous_audio_path",
        "next_audio_path",
        "previous_timestamp",
        "next_timestamp",
        "previous_label",
        "next_label",
        "output_audio_path",
        "severity_condition",
        "speaker_reference",
        "text_prompt",
    ]

    save_csv_rows(
        rows=rows,
        path=manifest_path,
        fieldnames=fieldnames,
    )


# ---------------------------------------------------------------------
# External command execution
# ---------------------------------------------------------------------


def build_external_command(
    command_template: str,
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Optional[str],
    external_repo_path: Optional[str],
    method_name: str,
) -> List[str]:
    """
    Build a shell command from a template.

    Supported placeholders:
        {manifest}
        {output_dir}
        {checkpoint}
        {repo}
        {method}

    Example template:
        "python external/qwen3_tts/generate.py --manifest {manifest} --output-dir {output_dir}"
    """

    filled = command_template.format(
        manifest=str(manifest_path),
        output_dir=str(output_dir),
        checkpoint=checkpoint_path or "",
        repo=external_repo_path or "",
        method=method_name,
    )

    return shlex.split(filled)


def run_external_command(
    command: List[str],
    cwd: Optional[str] = None,
    logger: Optional[Any] = None,
) -> int:
    """
    Run an external command and stream output.

    Returns
    -------
    int
        Process return code.
    """

    if logger is not None:
        logger.info(f"Running external command: {' '.join(command)}")
        if cwd is not None:
            logger.info(f"Working directory: {cwd}")

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        if logger is not None:
            logger.info(line.rstrip())
        else:
            print(line.rstrip())

    process.wait()
    return int(process.returncode)


# ---------------------------------------------------------------------
# Metadata validation and conversion
# ---------------------------------------------------------------------


def validate_generated_outputs(
    manifest_rows: List[Dict[str, Any]],
    require_audio: bool = True,
) -> Dict[str, Any]:
    """
    Check whether generated output files exist.
    """

    existing = 0
    missing = 0
    missing_paths: List[str] = []

    for row in manifest_rows:
        audio_path = Path(str(row["output_audio_path"]))

        if audio_path.exists():
            existing += 1
        else:
            missing += 1
            missing_paths.append(str(audio_path))

    if require_audio and missing > 0:
        raise FileNotFoundError(
            f"{missing} expected generated audio files are missing. "
            f"First missing file: {missing_paths[0]}"
        )

    return {
        "existing": existing,
        "missing": missing,
        "missing_paths": missing_paths[:20],
    }


def manifest_to_synthetic_metadata(
    manifest_rows: List[Dict[str, Any]],
    include_missing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert generation manifest rows into standardized synthetic metadata rows.
    """

    rows: List[Dict[str, Any]] = []

    for row in manifest_rows:
        audio_path = Path(str(row["output_audio_path"]))

        if not audio_path.exists() and not include_missing:
            continue

        rows.append(
            {
                "method": row["method"],
                "subject_id": row["subject_id"],
                "timestamp": row["timestamp"],
                "label": row["label"],
                "transcript": row["transcript"],
                "audio_path": str(audio_path) if audio_path.exists() else None,
                "reference_audio_path": row["reference_audio_path"],
                "previous_timestamp": row["previous_timestamp"],
                "next_timestamp": row["next_timestamp"],
                "previous_label": row["previous_label"],
                "next_label": row["next_label"],
            }
        )

    return rows


def save_synthetic_metadata(
    rows: List[Dict[str, Any]],
    metadata_path: Path,
) -> None:
    """
    Save standardized synthetic metadata.
    """

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
    ]

    save_csv_rows(
        rows=rows,
        path=metadata_path,
        fieldnames=fieldnames,
    )


# ---------------------------------------------------------------------
# Main external TTS baseline runner
# ---------------------------------------------------------------------


def run_external_tts_baseline(
    method_name: str,
    global_config: Dict[str, Any],
    method_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run an external TTS/pathology-TTS baseline through a standardized wrapper.

    Parameters
    ----------
    method_name:
        Name of the baseline, e.g., "qwen3_tts" or "scp_tts".
    global_config:
        Full baselines.yaml config.
    method_config:
        Method-specific config block.

    Returns
    -------
    dict
        Summary of manifest creation, external execution, and metadata saving.
    """

    experiment_cfg = global_config.get("experiment", {})
    output_cfg = global_config.get("output", {})

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    default_output_dir = (
        Path(experiment_cfg.get("output_dir", "outputs/generated_audio/baselines"))
        / method_name
    )

    output_dir = Path(method_config.get("output_dir", default_output_dir))
    ensure_dir(output_dir)

    logger = setup_logger(
        name=f"baselines.{method_name}",
        log_file=output_dir / f"{method_name}.log",
    )

    logger.info(f"Running external TTS baseline wrapper: {method_name}")
    logger.info(f"Output directory: {output_dir}")

    manifest_path = output_dir / "generation_manifest.csv"
    metadata_path = output_dir / output_cfg.get(
        "metadata_filename",
        "synthetic_metadata.csv",
    )

    manifest_rows = build_generation_manifest(
        method_name=method_name,
        global_config=global_config,
        method_config=method_config,
        output_dir=output_dir,
    )

    save_manifest(
        rows=manifest_rows,
        manifest_path=manifest_path,
    )

    logger.info(f"Saved generation manifest to {manifest_path}")
    logger.info(f"Number of requested samples: {len(manifest_rows)}")

    external_command = method_config.get("external_command")
    external_repo_path = method_config.get("external_repo_path")
    checkpoint_path = method_config.get("checkpoint_path")

    return_code: Optional[int] = None
    command_ran = False

    if external_command:
        command = build_external_command(
            command_template=external_command,
            manifest_path=manifest_path,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            external_repo_path=external_repo_path,
            method_name=method_name,
        )

        cwd = external_repo_path if external_repo_path else None

        return_code = run_external_command(
            command=command,
            cwd=cwd,
            logger=logger,
        )

        command_ran = True

        if return_code != 0:
            raise RuntimeError(
                f"External baseline command failed for {method_name} "
                f"with return code {return_code}."
            )

    else:
        logger.warning(
            "No external_command provided. "
            "Only the generation manifest was created. "
            "Run your external TTS system manually using this manifest, then rerun "
            "with generated files in place or convert the manifest to metadata."
        )

    require_audio = bool(method_config.get("require_audio", command_ran))

    validation = validate_generated_outputs(
        manifest_rows=manifest_rows,
        require_audio=require_audio,
    )

    metadata_rows = manifest_to_synthetic_metadata(
        manifest_rows=manifest_rows,
        include_missing=bool(method_config.get("include_missing_outputs", False)),
    )

    save_synthetic_metadata(
        rows=metadata_rows,
        metadata_path=metadata_path,
    )

    logger.info(f"Saved standardized metadata to {metadata_path}")
    logger.info(
        f"Generated audio existing={validation['existing']}, "
        f"missing={validation['missing']}"
    )

    summary = {
        "method": method_name,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "num_requested": len(manifest_rows),
        "num_metadata_rows": len(metadata_rows),
        "external_command_provided": external_command is not None,
        "command_ran": command_ran,
        "return_code": return_code,
        "validation": validation,
    }

    save_json(summary, output_dir / "summary.json")

    return summary


# ---------------------------------------------------------------------
# Optional helper for manual external generation workflow
# ---------------------------------------------------------------------


def convert_completed_manifest_to_metadata(
    manifest_path: str,
    metadata_path: str,
    include_missing: bool = False,
) -> Dict[str, Any]:
    """
    Convert a completed external generation manifest into synthetic metadata.

    This is useful when you manually run an external TTS repo outside this
    Python wrapper.

    Example
    -------
    After manually generating audio files listed in generation_manifest.csv:

        python -c "
        from baselines.tts_baselines import convert_completed_manifest_to_metadata
        convert_completed_manifest_to_metadata(
            'outputs/generated_audio/alstdi_baselines/qwen3_tts/generation_manifest.csv',
            'outputs/generated_audio/alstdi_baselines/qwen3_tts/synthetic_metadata.csv'
        )
        "
    """

    manifest_path_obj = Path(manifest_path)
    metadata_path_obj = Path(metadata_path)

    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

    df = pd.read_csv(manifest_path_obj)
    manifest_rows = df.to_dict(orient="records")

    validation = validate_generated_outputs(
        manifest_rows=manifest_rows,
        require_audio=not include_missing,
    )

    metadata_rows = manifest_to_synthetic_metadata(
        manifest_rows=manifest_rows,
        include_missing=include_missing,
    )

    save_synthetic_metadata(
        rows=metadata_rows,
        metadata_path=metadata_path_obj,
    )

    summary = {
        "manifest_path": str(manifest_path_obj),
        "metadata_path": str(metadata_path_obj),
        "num_manifest_rows": len(manifest_rows),
        "num_metadata_rows": len(metadata_rows),
        "validation": validation,
    }

    save_json(summary, metadata_path_obj.parent / "conversion_summary.json")

    return summary