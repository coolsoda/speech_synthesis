"""
Baseline augmentation methods for L-Flow.

This package contains:
- speed perturbation;
- VTLP;
- wrapper interfaces for external TTS/pathology-TTS baselines.
"""

from .speed_perturb import run_speed_perturbation
from .vtlp import run_vtlp
from .tts_baselines import (
    run_external_tts_baseline,
    convert_completed_manifest_to_metadata,
)

__all__ = [
    "run_speed_perturbation",
    "run_vtlp",
    "run_external_tts_baseline",
    "convert_completed_manifest_to_metadata",
]