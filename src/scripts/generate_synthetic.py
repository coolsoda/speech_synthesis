"""

Command-line script for generating synthetic longitudinal speech samples
using a trained L-Flow checkpoint.

Example usage
-------------
From the repository root:
    python src/scripts/generate_synthetic.py --config configs/inference.yaml

For ParkCeleb with command-line overrides:

    python src/scripts/generate_synthetic.py \
        --config configs/inference.yaml \
        --dataset-name parkceleb \
        --metadata-path data/metadata/parkceleb_metadata.csv \
        --split-path data/splits/parkceleb_split.json \
        --root-dir data/parkceleb \
        --checkpoint-path checkpoints/lflow_parkceleb/best.pt \
        --output-dir outputs/generated_audio/parkceleb_lflow \
        --temporal-resolution-days 800 \
        --max-samples-per-interval 3
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


from lflow.inference import generate_synthetic_dataset
from lflow.utils import load_yaml, save_json, update_nested


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic longitudinal speech samples with L-Flow."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to the inference YAML config file.",
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
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional override for checkpoint.lflow_path.",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="Optional override for checkpoint.vocoder_path.",
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
        "--num-steps",
        type=int,
        default=None,
        help="Optional override for generation.ode_solver.num_steps.",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=["euler", "heun"],
        help="Optional override for generation.ode_solver.name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for generation.batch_size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for generation.num_workers.",
    )

    parser.add_argument(
        "--target-num-frames",
        type=int,
        default=None,
        help="Optional override for generation.target_num_frames.",
    )

    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Force output.save_audio=true.",
    )

    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Force output.save_audio=false.",
    )

    parser.add_argument(
        "--save-mel",
        action="store_true",
        help="Force output.save_mel=true.",
    )

    parser.add_argument(
        "--no-save-mel",
        action="store_true",
        help="Force output.save_mel=false.",
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
        help="Force inference on CPU by setting device.use_cuda=false.",
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
        help="Optional path to save the returned generation summary JSON.",
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

    if args.metadata_path is not None:
        update_nested(config, ["dataset", "metadata_path"], args.metadata_path)

    if args.split_path is not None:
        update_nested(config, ["dataset", "split_path"], args.split_path)

    if args.root_dir is not None:
        update_nested(config, ["dataset", "root_dir"], args.root_dir)

    if args.generate_for_split is not None:
        update_nested(config, ["dataset", "generate_for_split"], args.generate_for_split)

    if args.checkpoint_path is not None:
        update_nested(config, ["checkpoint", "lflow_path"], args.checkpoint_path)

    if args.vocoder_path is not None:
        update_nested(config, ["checkpoint", "vocoder_path"], args.vocoder_path)

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

    if args.num_steps is not None:
        update_nested(config, ["generation", "ode_solver", "num_steps"], args.num_steps)

    if args.solver is not None:
        update_nested(config, ["generation", "ode_solver", "name"], args.solver)

    if args.batch_size is not None:
        update_nested(config, ["generation", "batch_size"], args.batch_size)

    if args.num_workers is not None:
        update_nested(config, ["generation", "num_workers"], args.num_workers)

    if args.target_num_frames is not None:
        update_nested(config, ["generation", "target_num_frames"], args.target_num_frames)

    if args.save_audio and args.no_save_audio:
        raise ValueError("Cannot use both --save-audio and --no-save-audio.")

    if args.save_audio:
        update_nested(config, ["output", "save_audio"], True)

    if args.no_save_audio:
        update_nested(config, ["output", "save_audio"], False)

    if args.save_mel and args.no_save_mel:
        raise ValueError("Cannot use both --save-mel and --no-save-mel.")

    if args.save_mel:
        update_nested(config, ["output", "save_mel"], True)

    if args.no_save_mel:
        update_nested(config, ["output", "save_mel"], False)

    if args.metadata_filename is not None:
        update_nested(config, ["output", "metadata_filename"], args.metadata_filename)

    if args.seed is not None:
        update_nested(config, ["experiment", "seed"], args.seed)

    if args.cpu:
        update_nested(config, ["device", "use_cuda"], False)

    if args.cuda_device is not None:
        update_nested(config, ["device", "cuda_device"], args.cuda_device)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Lightweight validation for required inference config fields.
    """

    required_fields = [
        ["dataset", "metadata_path"],
        ["dataset", "split_path"],
        ["checkpoint", "lflow_path"],
        ["generation", "temporal_resolution_days"],
        ["generation", "max_samples_per_interval"],
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
            "The paper setup generates synthetic samples only for training subjects."
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

    summary = generate_synthetic_dataset(config)

    if args.save_summary is not None:
        save_json(summary, args.save_summary)


if __name__ == "__main__":
    main()