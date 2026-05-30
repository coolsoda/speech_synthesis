"""

This script trains downstream regressors on:

    D_aug = D_real ∪ D_syn

and evaluates on held-out real test subjects.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict



REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from lflow.utils import load_yaml, save_json, update_nested


# This module will be implemented under:
#     src/evaluation/downstream_regression.py
from evaluation.downstream_regression import run_downstream_evaluation


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run downstream regression evaluation for L-Flow augmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/downstream.yaml",
        help="Path to downstream YAML config.",
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
        "--dataset-name",
        type=str,
        default=None,
        choices=["alstdi", "parkceleb"],
        help="Optional override for dataset.name.",
    )

    parser.add_argument(
        "--real-metadata-path",
        type=str,
        default=None,
        help="Optional override for dataset.real_metadata_path.",
    )

    parser.add_argument(
        "--synthetic-metadata-path",
        type=str,
        default=None,
        help="Optional override for dataset.synthetic_metadata_path.",
    )

    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Optional override for dataset.split_path.",
    )

    parser.add_argument(
        "--train-with-synthetic",
        action="store_true",
        help="Force dataset.train_with_synthetic=true.",
    )

    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Force dataset.train_with_synthetic=false.",
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        default=None,
        choices=["audio_embedding", "handcrafted"],
        help="Optional override for features.type.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Optional override for features.audio_embedding.checkpoint, "
            "for example facebook/wav2vec2-base."
        ),
    )

    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default=None,
        help="Optional override for the selected feature cache directory.",
    )

    parser.add_argument(
        "--disable-ridge",
        action="store_true",
        help="Disable Ridge Regression.",
    )

    parser.add_argument(
        "--disable-xgboost",
        action="store_true",
        help="Disable XGBoost Regressor.",
    )

    parser.add_argument(
        "--disable-mlp",
        action="store_true",
        help="Disable MLP Regressor.",
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
        help="Force evaluation on CPU by setting device.use_cuda=false.",
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
        help="Optional path to save the returned downstream summary JSON.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Config override helpers
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

    if args.dataset_name is not None:
        update_nested(config, ["dataset", "name"], args.dataset_name)

    if args.real_metadata_path is not None:
        update_nested(
            config,
            ["dataset", "real_metadata_path"],
            args.real_metadata_path,
        )

    if args.synthetic_metadata_path is not None:
        update_nested(
            config,
            ["dataset", "synthetic_metadata_path"],
            args.synthetic_metadata_path,
        )

    if args.split_path is not None:
        update_nested(config, ["dataset", "split_path"], args.split_path)

    if args.train_with_synthetic and args.real_only:
        raise ValueError("Cannot use both --train-with-synthetic and --real-only.")

    if args.train_with_synthetic:
        update_nested(config, ["dataset", "train_with_synthetic"], True)

    if args.real_only:
        update_nested(config, ["dataset", "train_with_synthetic"], False)

    if args.feature_type is not None:
        update_nested(config, ["features", "type"], args.feature_type)

    if args.embedding_model is not None:
        update_nested(
            config,
            ["features", "audio_embedding", "checkpoint"],
            args.embedding_model,
        )

    if args.feature_cache_dir is not None:
        feature_type = config.get("features", {}).get("type", "audio_embedding")

        if feature_type == "audio_embedding":
            update_nested(
                config,
                ["features", "audio_embedding", "cache_dir"],
                args.feature_cache_dir,
            )
        elif feature_type == "handcrafted":
            update_nested(
                config,
                ["features", "handcrafted", "cache_dir"],
                args.feature_cache_dir,
            )

    if args.disable_ridge:
        update_nested(config, ["regressors", "ridge", "enabled"], False)

    if args.disable_xgboost:
        update_nested(config, ["regressors", "xgboost", "enabled"], False)

    if args.disable_mlp:
        update_nested(config, ["regressors", "mlp", "enabled"], False)

    if args.seed is not None:
        update_nested(config, ["experiment", "seed"], args.seed)

    if args.cpu:
        update_nested(config, ["device", "use_cuda"], False)

    if args.cuda_device is not None:
        update_nested(config, ["device", "cuda_device"], args.cuda_device)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Lightweight validation for required downstream config fields.
    """

    required_fields = [
        ["dataset", "real_metadata_path"],
        ["dataset", "split_path"],
        ["dataset", "subject_column"],
        ["dataset", "audio_column"],
        ["dataset", "label_column"],
        ["features", "type"],
        ["regressors"],
    ]

    for keys in required_fields:
        current: Any = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                dotted = ".".join(keys)
                raise ValueError(f"Missing required config field: {dotted}")
            current = current[key]

    train_with_synthetic = bool(
        config.get("dataset", {}).get("train_with_synthetic", True)
    )

    synthetic_path = config.get("dataset", {}).get("synthetic_metadata_path")

    if train_with_synthetic and not synthetic_path:
        raise ValueError(
            "dataset.train_with_synthetic=true, but "
            "dataset.synthetic_metadata_path is missing."
        )


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

    summary = run_downstream_evaluation(config)

    if args.save_summary is not None:
        save_json(summary, args.save_summary)


if __name__ == "__main__":
    main()