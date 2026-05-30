"""

Speech validity evaluation for generated synthetic speech.

This module evaluates:
1. Intelligibility using Word Error Rate (WER).
2. Speaker preservation using speaker similarity (SIM).

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.spatial.distance import cosine

from lflow.utils import ensure_dir, save_csv_rows, save_json, setup_logger


# ---------------------------------------------------------------------
# Text normalization and WER
# ---------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """
    Normalize text before WER computation.
    """

    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """
    Compute Levenshtein edit distance between two token sequences.
    """

    n = len(ref)
    m = len(hyp)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i, 0] = i

    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return int(dp[n, m])


def compute_wer(reference_text: Any, hypothesis_text: Any) -> float:
    """
    Compute word error rate.

    WER = edit_distance(reference_words, hypothesis_words) / len(reference_words)
    """

    ref = normalize_text(reference_text).split()
    hyp = normalize_text(hypothesis_text).split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    distance = levenshtein_distance(ref, hyp)
    return float(distance / len(ref))


# ---------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    max_seconds: Optional[float] = None,
) -> torch.Tensor:
    """
    Load audio as mono waveform.

    Returns
    -------
    waveform:
        Tensor with shape [T].
    """

    path = Path(str(audio_path))

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=sample_rate,
        )

    if max_seconds is not None:
        max_samples = int(sample_rate * max_seconds)
        waveform = waveform[:max_samples]

    max_abs = waveform.abs().max()
    if max_abs > 0:
        waveform = waveform / max_abs

    return waveform


# ---------------------------------------------------------------------
# ASR wrappers
# ---------------------------------------------------------------------


class BaseASRBackend:
    """
    Base interface for ASR backends.
    """

    name: str = "base_asr"

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class HuggingFaceASRBackend(BaseASRBackend):
    """
    Hugging Face ASR pipeline wrapper.

    Example models:
        - openai/whisper-large-v3
        - facebook/hubert-large-ls960-ft
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        chunk_length_s: Optional[float] = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HuggingFaceASRBackend. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_name = model_name
        self.name = name or model_name.replace("/", "_")
        self.device = device or torch.device("cpu")

        if self.device.type == "cuda":
            device_arg = self.device.index if self.device.index is not None else 0
        else:
            device_arg = -1

        kwargs = {
            "task": "automatic-speech-recognition",
            "model": model_name,
            "device": device_arg,
        }

        if chunk_length_s is not None:
            kwargs["chunk_length_s"] = chunk_length_s

        self.pipe = pipeline(**kwargs)

    def transcribe(self, audio_path: str) -> str:
        output = self.pipe(audio_path)

        if isinstance(output, dict):
            return str(output.get("text", ""))

        return str(output)


class PrecomputedASRBackend(BaseASRBackend):
    """
    ASR backend from a precomputed CSV.

    Expected CSV columns:
        audio_path,transcript

    This is useful when ASR outputs are generated externally by models such as
    Qwen3-ASR or VibeVoice-ASR.
    """

    def __init__(
        self,
        csv_path: str,
        name: str = "precomputed_asr",
        audio_column: str = "audio_path",
        transcript_column: str = "transcript",
    ) -> None:
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"Precomputed ASR CSV not found: {path}")

        df = pd.read_csv(path)

        if audio_column not in df.columns or transcript_column not in df.columns:
            raise ValueError(
                f"Precomputed ASR CSV must contain {audio_column} and "
                f"{transcript_column} columns."
            )

        self.name = name
        self.audio_column = audio_column
        self.transcript_column = transcript_column

        self.mapping = {
            str(row[audio_column]): str(row[transcript_column])
            for _, row in df.iterrows()
        }

    def transcribe(self, audio_path: str) -> str:
        audio_path = str(audio_path)

        if audio_path in self.mapping:
            return self.mapping[audio_path]

        # Try basename fallback.
        basename = Path(audio_path).name

        for key, value in self.mapping.items():
            if Path(key).name == basename:
                return value

        raise KeyError(f"No precomputed ASR output found for audio: {audio_path}")


# ---------------------------------------------------------------------
# Speaker embedding wrappers
# ---------------------------------------------------------------------


class BaseSpeakerBackend:
    """
    Base interface for speaker embedding backends.
    """

    name: str = "base_speaker"

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        raise NotImplementedError


class SpeechBrainSpeakerBackend(BaseSpeakerBackend):
    """
    SpeechBrain speaker embedding wrapper.

    Example source:
        speechbrain/spkrec-ecapa-voxceleb

    Install:
        pip install speechbrain
    """

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
    ) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for SpeechBrainSpeakerBackend. "
                "Install it with: pip install speechbrain"
            ) from exc

        self.source = source
        self.savedir = savedir
        self.name = name or source.replace("/", "_")
        self.device = device or torch.device("cpu")

        run_opts = {"device": str(self.device)}

        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts=run_opts,
        )

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        signal = self.classifier.load_audio(audio_path)
        embedding = self.classifier.encode_batch(signal)

        embedding = embedding.squeeze().detach().cpu().numpy()
        return embedding.astype(np.float32)


class WavLMSpeakerBackend(BaseSpeakerBackend):
    """
    Generic Hugging Face SSL speaker/audio embedding wrapper.

    This is not a dedicated speaker verification model by itself, but it can
    be useful as a fallback embedding extractor.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        pooling: str = "mean",
        name: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for WavLMSpeakerBackend. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_name = model_name
        self.name = name or model_name.replace("/", "_")
        self.device = device or torch.device("cpu")
        self.sample_rate = sample_rate
        self.pooling = pooling

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        waveform = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            max_seconds=None,
        )

        waveform = waveform.unsqueeze(0).to(self.device)

        output = self.model(input_values=waveform)
        hidden = output.last_hidden_state

        if self.pooling == "mean":
            emb = hidden.mean(dim=1)
        elif self.pooling == "max":
            emb = hidden.max(dim=1).values
        elif self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


class PrecomputedSpeakerBackend(BaseSpeakerBackend):
    """
    Speaker backend from precomputed embeddings.

    Expected file format:
        .npz containing:
            keys: audio_path strings or sanitized identifiers
            values: embedding arrays

    This is useful for speaker models such as WeSpeaker or TitaNet when their
    embeddings are computed externally.
    """

    def __init__(
        self,
        npz_path: str,
        name: str = "precomputed_speaker",
    ) -> None:
        path = Path(npz_path)

        if not path.exists():
            raise FileNotFoundError(f"Precomputed speaker embeddings not found: {path}")

        self.name = name
        self.data = np.load(path, allow_pickle=True)

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        key = str(audio_path)

        if key in self.data:
            return np.asarray(self.data[key], dtype=np.float32)

        basename = Path(audio_path).name

        if basename in self.data:
            return np.asarray(self.data[basename], dtype=np.float32)

        safe = sanitize_key(audio_path)

        if safe in self.data:
            return np.asarray(self.data[safe], dtype=np.float32)

        raise KeyError(f"No precomputed speaker embedding found for {audio_path}")


def sanitize_key(path: str) -> str:
    """
    Create a safe key for precomputed embedding storage.
    """

    return str(path).replace("/", "_").replace("\\", "_").replace(":", "_")


# ---------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity safely.
    """

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < eps:
        return 0.0

    return float(np.dot(a, b) / denom)


def compute_speaker_similarity(
    generated_audio_path: str,
    reference_audio_path: str,
    backend: BaseSpeakerBackend,
) -> float:
    """
    Compute speaker similarity between generated and reference audio.
    """

    emb_gen = backend.extract_embedding(generated_audio_path)
    emb_ref = backend.extract_embedding(reference_audio_path)

    return cosine_similarity(emb_gen, emb_ref)


# ---------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------


def get_device(use_cuda: bool = True, cuda_device: int = 0) -> torch.device:
    """
    Get torch device.
    """

    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")

    return torch.device("cpu")


def build_asr_backends(
    config: Dict[str, Any],
    device: torch.device,
) -> List[BaseASRBackend]:
    """
    Build ASR backends from config.

    Example config:
        asr:
          backends:
            - type: huggingface
              name: whisper
              model_name: openai/whisper-large-v3
            - type: precomputed
              name: qwen3_asr
              csv_path: outputs/asr/qwen3_asr.csv
    """

    asr_cfg = config.get("asr", {})
    backend_cfgs = asr_cfg.get("backends", [])

    backends: List[BaseASRBackend] = []

    for cfg in backend_cfgs:
        backend_type = cfg.get("type", "huggingface")

        if not cfg.get("enabled", True):
            continue

        if backend_type == "huggingface":
            backends.append(
                HuggingFaceASRBackend(
                    model_name=cfg["model_name"],
                    device=device,
                    name=cfg.get("name"),
                    chunk_length_s=cfg.get("chunk_length_s"),
                )
            )

        elif backend_type == "precomputed":
            backends.append(
                PrecomputedASRBackend(
                    csv_path=cfg["csv_path"],
                    name=cfg.get("name", "precomputed_asr"),
                    audio_column=cfg.get("audio_column", "audio_path"),
                    transcript_column=cfg.get("transcript_column", "transcript"),
                )
            )

        else:
            raise ValueError(f"Unknown ASR backend type: {backend_type}")

    return backends


def build_speaker_backends(
    config: Dict[str, Any],
    device: torch.device,
) -> List[BaseSpeakerBackend]:
    """
    Build speaker embedding backends from config.

    Example config:
        speaker:
          backends:
            - type: speechbrain
              name: ecapa
              source: speechbrain/spkrec-ecapa-voxceleb
            - type: wavlm
              name: wavlm
              model_name: microsoft/wavlm-base-plus
            - type: precomputed
              name: titanet
              npz_path: outputs/speaker/titanet_embeddings.npz
    """

    speaker_cfg = config.get("speaker", {})
    backend_cfgs = speaker_cfg.get("backends", [])

    backends: List[BaseSpeakerBackend] = []

    for cfg in backend_cfgs:
        backend_type = cfg.get("type", "speechbrain")

        if not cfg.get("enabled", True):
            continue

        if backend_type == "speechbrain":
            backends.append(
                SpeechBrainSpeakerBackend(
                    source=cfg.get("source", "speechbrain/spkrec-ecapa-voxceleb"),
                    savedir=cfg.get("savedir", "pretrained_models/spkrec-ecapa-voxceleb"),
                    device=device,
                    name=cfg.get("name"),
                )
            )

        elif backend_type == "wavlm":
            backends.append(
                WavLMSpeakerBackend(
                    model_name=cfg.get("model_name", "microsoft/wavlm-base-plus"),
                    device=device,
                    sample_rate=int(cfg.get("sample_rate", 16000)),
                    pooling=cfg.get("pooling", "mean"),
                    name=cfg.get("name"),
                )
            )

        elif backend_type == "precomputed":
            backends.append(
                PrecomputedSpeakerBackend(
                    npz_path=cfg["npz_path"],
                    name=cfg.get("name", "precomputed_speaker"),
                )
            )

        else:
            raise ValueError(f"Unknown speaker backend type: {backend_type}")

    return backends


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def validate_metadata(df: pd.DataFrame) -> None:
    """
    Validate synthetic metadata.
    """

    required = [
        "subject_id",
        "audio_path",
        "transcript",
        "reference_audio_path",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Synthetic metadata is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def evaluate_wer(
    df: pd.DataFrame,
    asr_backends: List[BaseASRBackend],
    logger: Any,
) -> List[Dict[str, Any]]:
    """
    Evaluate WER for all rows and ASR backends.
    """

    rows: List[Dict[str, Any]] = []

    if len(asr_backends) == 0:
        logger.warning("No ASR backends configured. Skipping WER evaluation.")
        return rows

    for backend in asr_backends:
        logger.info(f"Running WER evaluation with ASR backend: {backend.name}")

        for i, row in df.iterrows():
            audio_path = str(row["audio_path"])
            reference_text = row.get("transcript", "")

            try:
                hypothesis = backend.transcribe(audio_path)
                wer = compute_wer(reference_text, hypothesis)

                rows.append(
                    {
                        "metric": "wer",
                        "backend": backend.name,
                        "method": row.get("method", "unknown"),
                        "subject_id": row["subject_id"],
                        "audio_path": audio_path,
                        "reference_text": reference_text,
                        "hypothesis_text": hypothesis,
                        "value": wer,
                    }
                )

            except Exception as exc:
                logger.warning(
                    f"WER failed for backend={backend.name}, audio={audio_path}: {exc}"
                )

    return rows


def evaluate_speaker_similarity(
    df: pd.DataFrame,
    speaker_backends: List[BaseSpeakerBackend],
    logger: Any,
) -> List[Dict[str, Any]]:
    """
    Evaluate speaker similarity for all rows and speaker backends.
    """

    rows: List[Dict[str, Any]] = []

    if len(speaker_backends) == 0:
        logger.warning("No speaker backends configured. Skipping SIM evaluation.")
        return rows

    for backend in speaker_backends:
        logger.info(f"Running speaker similarity with backend: {backend.name}")

        for i, row in df.iterrows():
            audio_path = str(row["audio_path"])
            reference_audio_path = str(row["reference_audio_path"])

            try:
                sim = compute_speaker_similarity(
                    generated_audio_path=audio_path,
                    reference_audio_path=reference_audio_path,
                    backend=backend,
                )

                rows.append(
                    {
                        "metric": "sim",
                        "backend": backend.name,
                        "method": row.get("method", "unknown"),
                        "subject_id": row["subject_id"],
                        "audio_path": audio_path,
                        "reference_audio_path": reference_audio_path,
                        "value": sim,
                    }
                )

            except Exception as exc:
                logger.warning(
                    f"SIM failed for backend={backend.name}, audio={audio_path}: {exc}"
                )

    return rows


def aggregate_metric_rows(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metric rows by metric, backend, and method.
    """

    if len(rows) == 0:
        return {}

    df = pd.DataFrame(rows)

    summary: Dict[str, Any] = {}

    group_cols = ["metric", "backend", "method"]

    for keys, group in df.groupby(group_cols):
        metric, backend, method = keys

        values = group["value"].astype(float).to_numpy()

        key = f"{metric}/{backend}/{method}"

        summary[key] = {
            "metric": metric,
            "backend": backend,
            "method": method,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "num_samples": int(len(values)),
        }

    return summary


def run_speech_validity_evaluation(
    synthetic_metadata_path: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run speech validity evaluation.

    Parameters
    ----------
    synthetic_metadata_path:
        Path to synthetic metadata CSV.

    output_dir:
        Directory where evaluation outputs will be saved.

    config:
        Optional config dictionary for ASR and speaker backends.

    Returns
    -------
    dict
        Summary of WER and SIM evaluation.
    """

    config = config or {}

    metadata_path = Path(synthetic_metadata_path)
    output_path = Path(output_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Synthetic metadata not found: {metadata_path}")

    ensure_dir(output_path)

    logger = setup_logger(
        name="evaluation.speech_validity",
        log_file=output_path / "speech_validity.log",
    )

    logger.info(f"Loading synthetic metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)
    validate_metadata(df)

    # Drop rows without generated audio.
    df = df[df["audio_path"].notna()].copy()
    df = df[df["audio_path"].astype(str) != "None"].copy()

    logger.info(f"Number of generated samples to evaluate: {len(df)}")

    device_cfg = config.get("device", {})
    device = get_device(
        use_cuda=bool(device_cfg.get("use_cuda", True)),
        cuda_device=int(device_cfg.get("cuda_device", 0)),
    )

    asr_backends = build_asr_backends(config, device=device)
    speaker_backends = build_speaker_backends(config, device=device)

    all_rows: List[Dict[str, Any]] = []

    if config.get("evaluate_wer", True):
        wer_rows = evaluate_wer(
            df=df,
            asr_backends=asr_backends,
            logger=logger,
        )
        all_rows.extend(wer_rows)

    if config.get("evaluate_sim", True):
        sim_rows = evaluate_speaker_similarity(
            df=df,
            speaker_backends=speaker_backends,
            logger=logger,
        )
        all_rows.extend(sim_rows)

    detailed_path = output_path / "speech_validity_detailed.csv"
    summary_path = output_path / "speech_validity_summary.json"

    fieldnames = [
        "metric",
        "backend",
        "method",
        "subject_id",
        "audio_path",
        "reference_audio_path",
        "reference_text",
        "hypothesis_text",
        "value",
    ]

    save_csv_rows(
        rows=all_rows,
        path=detailed_path,
        fieldnames=fieldnames,
    )

    summary = {
        "synthetic_metadata_path": str(metadata_path),
        "output_dir": str(output_path),
        "num_samples": int(len(df)),
        "num_metric_rows": int(len(all_rows)),
        "aggregated": aggregate_metric_rows(all_rows),
        "detailed_path": str(detailed_path),
    }

    save_json(summary, summary_path)

    logger.info(f"Saved detailed metrics to {detailed_path}")
    logger.info(f"Saved summary to {summary_path}")

    return summary


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------


def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load JSON config for speech validity evaluation.

    YAML is intentionally avoided here to keep this standalone evaluator simple.
    """

    if path is None:
        return {}

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Config JSON not found: {path_obj}")

    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate speech validity of generated synthetic speech."
    )

    parser.add_argument(
        "--synthetic-metadata",
        type=str,
        required=True,
        help="Path to synthetic_metadata.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save speech validity metrics.",
    )

    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=(
            "Optional JSON config defining ASR and speaker backends. "
            "If omitted, no backends are run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json_config(args.config_json)

    run_speech_validity_evaluation(
        synthetic_metadata_path=args.synthetic_metadata,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()