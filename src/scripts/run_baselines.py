"""

Command-line script for running baseline augmentation methods.

This script generates synthetic training samples using baseline
methods defined in configs/baselines.yaml.

Baseline categories:
    1. Signal-level augmentation
        - Speed perturbation
        - VTLP

    2. Modern TTS systems
        - OmniVoice
        - Qwen3-TTS
        - VoxCPM2
        - XTTS-v2

    3. Pathology-oriented TTS systems
        - TTDS
        - SCP-TTS

Example usage
-------------
From the repository root:

    python src/scripts/run_baselines.py --config configs/baselines.yaml

Run only signal-level baselines:

    python src/scripts/run_baselines.py \
        --config configs/baselines.yaml \
        --only-signal-level
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from lflow.utils import load_yaml, save_json, update_nested


# These modules will be implemented under src/baselines/.
from baselines.speed_perturb import run_speed_perturbation
from baselines.vtlp import run_vtlp
from baselines.tts_baselines import run_external_tts_baseline


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline augmentation methods."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baselines.yaml",
        help="Path to baseline YAML config.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional override for experiment.name.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for experiment.output_dir.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional override for experiment.log_dir.",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        choices=["alstdi", "parkceleb"],
        help="Optional override for dataset.name.",
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional override for dataset.metadata_path.",
    )

    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Optional override for dataset.split_path.",
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Optional override for dataset.root_dir.",
    )

    parser.add_argument(
        "--generate-for-split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help=(
            "Optional override for dataset.generate_for_split. "
            "For the paper setup, this should normally be 'train'."
        ),
    )

    parser.add_argument(
        "--temporal-resolution-days",
        type=float,
        default=None,
        help=(
            "Optional override for generation.temporal_resolution_days. "
            "Paper defaults: ALS-TDI=30, ParkCeleb=800."
        ),
    )

    parser.add_argument(
        "--max-samples-per-interval",
        type=int,
        default=None,
        help=(
            "Optional override for generation.max_samples_per_interval. "
            "Paper defaults: ALS-TDI=2, ParkCeleb=3."
        ),
    )

    parser.add_argument(
        "--only-signal-level",
        action="store_true",
        help="Run only speed perturbation and VTLP.",
    )

    parser.add_argument(
        "--only-tts",
        action="store_true",
        help="Run only modern/pathology TTS wrapper baselines.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of methods to run. "
            "Examples: speed_perturbation vtlp qwen3_tts scp_tts"
        ),
    )

    parser.add_argument(
        "--enable-all",
        action="store_true",
        help=(
            "Force all baselines to enabled=true. "
            "Use carefully because external TTS baselines require checkpoints/repos."
        ),
    )

    parser.add_argument(
        "--metadata-filename",
        type=str,
        default=None,
        help="Optional override for output.metadata_filename.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for experiment.seed.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution by setting device.use_cuda=false.",
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Optional override for device.cuda_device.",
    )

    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional path to save the returned baseline summary JSON.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------


def apply_cli_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Apply command-line overrides to a loaded YAML config.
    """

    if args.experiment_name is not None:
        update_nested(config, ["experiment", "name"], args.experiment_name)

    if args.output_dir is not None:
        update_nested(config, ["experiment", "output_dir"], args.output_dir)

    if args.log_dir is not None:
        update_nested(config, ["experiment", "log_dir"], args.log_dir)

    if args.dataset_name is not None:
        update_nested(config, ["dataset", "name"], args.dataset_name)

    if args.metadata_path is not None:
        update_nested(config, ["dataset", "metadata_path"], args.metadata_path)

    if args.split_path is not None:
        update_nested(config, ["dataset", "split_path"], args.split_path)

    if args.root_dir is not None:
        update_nested(config, ["dataset", "root_dir"], args.root_dir)

    if args.generate_for_split is not None:
        update_nested(config, ["dataset", "generate_for_split"], args.generate_for_split)

    if args.temporal_resolution_days is not None:
        update_nested(
            config,
            ["generation", "temporal_resolution_days"],
            args.temporal_resolution_days,
        )

    if args.max_samples_per_interval is not None:
        update_nested(
            config,
            ["generation", "max_samples_per_interval"],
            args.max_samples_per_interval,
        )

    if args.metadata_filename is not None:
        update_nested(config, ["output", "metadata_filename"], args.metadata_filename)

    if args.seed is not None:
        update_nested(config, ["experiment", "seed"], args.seed)

    if args.cpu:
        update_nested(config, ["device", "use_cuda"], False)

    if args.cuda_device is not None:
        update_nested(config, ["device", "cuda_device"], args.cuda_device)

    if args.enable_all:
        enable_all_baselines(config)

    return config


def enable_all_baselines(config: Dict[str, Any]) -> None:
    """
    Set enabled=true for all baselines in the config.
    """

    baseline_cfg = config.get("baselines", {})

    for group in baseline_cfg.values():
        if not isinstance(group, dict):
            continue

        for method_cfg in group.values():
            if isinstance(method_cfg, dict):
                method_cfg["enabled"] = True


def validate_config(config: Dict[str, Any]) -> None:
    """
    Lightweight validation for required baseline config fields.
    """

    required_fields = [
        ["dataset", "metadata_path"],
        ["dataset", "split_path"],
        ["generation", "temporal_resolution_days"],
        ["generation", "max_samples_per_interval"],
        ["baselines"],
    ]

    for keys in required_fields:
        current: Any = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                dotted = ".".join(keys)
                raise ValueError(f"Missing required config field: {dotted}")
            current = current[key]

    generate_for_split = config.get("dataset", {}).get("generate_for_split", "train")

    if generate_for_split != "train":
        print(
            "Warning: dataset.generate_for_split is not 'train'. "
            "The paper setup generates baseline samples only for training subjects."
        )


# ---------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------


def method_selected(
    method_name: str,
    args: argparse.Namespace,
    category: str,
) -> bool:
    """
    Decide whether a method should run based on CLI selection flags.
    """

    if args.methods is not None:
        return method_name in set(args.methods)

    if args.only_signal_level:
        return category == "signal_level"

    if args.only_tts:
        return category in {"modern_tts", "pathology_tts"}

    return True


def get_enabled_methods(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Collect enabled baseline methods from config.
    """

    baseline_cfg = config.get("baselines", {})

    methods: List[Dict[str, Any]] = []

    for category, category_cfg in baseline_cfg.items():
        if not isinstance(category_cfg, dict):
            continue

        for method_name, method_cfg in category_cfg.items():
            if not isinstance(method_cfg, dict):
                continue

            if not method_cfg.get("enabled", False):
                continue

            if not method_selected(method_name, args, category):
                continue

            methods.append(
                {
                    "category": category,
                    "name": method_name,
                    "config": method_cfg,
                }
            )

    return methods


# ---------------------------------------------------------------------
# Main baseline runner
# ---------------------------------------------------------------------


def run_baselines(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run all selected baseline augmentation methods.

    Returns
    -------
    dict
        Summary of baseline generation.
    """

    methods = get_enabled_methods(config, args)

    if len(methods) == 0:
        raise ValueError(
            "No baseline methods selected. Check baselines.yaml or CLI flags."
        )

    summary: Dict[str, Any] = {
        "num_methods": len(methods),
        "methods": {},
    }

    for method in methods:
        category = method["category"]
        method_name = method["name"]
        method_cfg = method["config"]

        print(f"\nRunning baseline: {method_name} ({category})")

        if method_name == "speed_perturbation":
            result = run_speed_perturbation(
                global_config=config,
                method_config=method_cfg,
            )

        elif method_name == "vtlp":
            result = run_vtlp(
                global_config=config,
                method_config=method_cfg,
            )

        else:
            result = run_external_tts_baseline(
                method_name=method_name,
                global_config=config,
                method_config=method_cfg,
            )

        summary["methods"][method_name] = {
            "category": category,
            "result": result,
        }

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml(config_path)
    config = apply_cli_overrides(config, args)
    validate_config(config)

    summary = run_baselines(config, args)

    if args.save_summary is not None:
        save_json(summary, args.save_summary)


if __name__ == "__main__":
    main()