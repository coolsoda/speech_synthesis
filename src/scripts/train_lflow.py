"""

Command-line script for training L-Flow.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Make src/ importable when running this script directly
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from lflow.train import train_lflow
from lflow.utils import load_yaml, save_json, update_nested


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train L-Flow from a YAML configuration file."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training YAML config file.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint path to resume from. "
            "Use 'latest' to resume from the most recent checkpoint in checkpoint_dir."
        ),
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
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional override for experiment.checkpoint_dir.",
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
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training.epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for training.batch_size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional override for training.optimizer.lr.",
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
        help="Force training on CPU by setting device.use_cuda=false.",
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Optional override for device.cuda_device.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for training.num_workers.",
    )

    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional path to save the returned training summary JSON.",
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

    if args.resume is not None:
        update_nested(config, ["experiment", "resume_from"], args.resume)

    if args.experiment_name is not None:
        update_nested(config, ["experiment", "name"], args.experiment_name)

    if args.output_dir is not None:
        update_nested(config, ["experiment", "output_dir"], args.output_dir)

    if args.checkpoint_dir is not None:
        update_nested(config, ["experiment", "checkpoint_dir"], args.checkpoint_dir)

    if args.dataset_name is not None:
        update_nested(config, ["dataset", "name"], args.dataset_name)

    if args.metadata_path is not None:
        update_nested(config, ["dataset", "metadata_path"], args.metadata_path)

    if args.split_path is not None:
        update_nested(config, ["dataset", "split_path"], args.split_path)

    if args.root_dir is not None:
        update_nested(config, ["dataset", "root_dir"], args.root_dir)

    if args.epochs is not None:
        update_nested(config, ["training", "epochs"], args.epochs)

    if args.batch_size is not None:
        update_nested(config, ["training", "batch_size"], args.batch_size)

    if args.lr is not None:
        update_nested(config, ["training", "optimizer", "lr"], args.lr)

    if args.seed is not None:
        update_nested(config, ["experiment", "seed"], args.seed)

    if args.cpu:
        update_nested(config, ["device", "use_cuda"], False)

    if args.cuda_device is not None:
        update_nested(config, ["device", "cuda_device"], args.cuda_device)

    if args.num_workers is not None:
        update_nested(config, ["training", "num_workers"], args.num_workers)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Lightweight validation for required training config fields.
    """

    required_paths = [
        ["dataset", "metadata_path"],
        ["dataset", "split_path"],
    ]

    for keys in required_paths:
        current: Any = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                dotted = ".".join(keys)
                raise ValueError(f"Missing required config field: {dotted}")
            current = current[key]

    if "training" not in config:
        raise ValueError("Missing required config section: training")

    if "model" not in config:
        raise ValueError("Missing required config section: model")


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

    summary = train_lflow(config)

    if args.save_summary is not None:
        save_json(summary, args.save_summary)


if __name__ == "__main__":
    main()