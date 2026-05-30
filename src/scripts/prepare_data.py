"""

The script supports:
1. Validating required metadata columns.
2. Resolving audio paths.
3. Checking whether audio files exist.
4. Parsing timestamps.
5. Normalizing clinical labels into [0, 1].
6. Creating subject-level train/test splits.
7. Saving cleaned metadata and split JSON.

Expected input metadata columns by default:
    subject_id,audio_path,transcript,timestamp,label

Example usage
-------------
From the repository root:

    python src/scripts/prepare_data.py \
        --input-metadata data/raw/alstdi_metadata_raw.csv \
        --output-metadata data/metadata/alstdi_metadata.csv \
        --output-split data/splits/alstdi_split.json \
        --root-dir data/alstdi \
        --dataset-name alstdi \
        --label-mode alstdi \
        --test-size 0.2

For ParkCeleb:

    python src/scripts/prepare_data.py \
        --input-metadata data/raw/parkceleb_metadata_raw.csv \
        --output-metadata data/metadata/parkceleb_metadata.csv \
        --output-split data/splits/parkceleb_split.json \
        --root-dir data/parkceleb \
        --dataset-name parkceleb \
        --label-mode parkceleb \
        --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from lflow.utils import ensure_dir, save_json, set_seed


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate longitudinal speech metadata for L-Flow."
    )

    parser.add_argument(
        "--input-metadata",
        type=str,
        required=True,
        help="Path to raw input metadata CSV.",
    )

    parser.add_argument(
        "--output-metadata",
        type=str,
        required=True,
        help="Path to save cleaned metadata CSV.",
    )

    parser.add_argument(
        "--output-split",
        type=str,
        required=True,
        help="Path to save subject-level split JSON.",
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help=(
            "Optional dataset root directory. Relative audio paths will be "
            "resolved against this directory when checking file existence."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom",
        choices=["alstdi", "parkceleb", "custom"],
        help="Dataset name used for summary metadata.",
    )

    parser.add_argument(
        "--subject-column",
        type=str,
        default="subject_id",
        help="Column name for subject IDs.",
    )

    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio_path",
        help="Column name for audio file paths.",
    )

    parser.add_argument(
        "--transcript-column",
        type=str,
        default="transcript",
        help="Column name for transcripts or text prompts.",
    )

    parser.add_argument(
        "--time-column",
        type=str,
        default="timestamp",
        help="Column name for recording timestamps.",
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for raw clinical labels.",
    )

    parser.add_argument(
        "--label-mode",
        type=str,
        default="none",
        choices=["none", "minmax", "alstdi", "parkceleb"],
        help=(
            "How to normalize labels. "
            "'alstdi' maps ALSFRS-R speech subscore y to (4-y)/4. "
            "'parkceleb' maps pre/post diagnosis labels to 0/1."
        ),
    )

    parser.add_argument(
        "--pre-diagnosis-values",
        type=str,
        nargs="*",
        default=["pre", "pre-diagnosis", "before", "0", "false", "False"],
        help="Values treated as pre-diagnosis for ParkCeleb label normalization.",
    )

    parser.add_argument(
        "--post-diagnosis-values",
        type=str,
        nargs="*",
        default=["post", "post-diagnosis", "after", "1", "true", "True"],
        help="Values treated as post-diagnosis for ParkCeleb label normalization.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of subjects assigned to the test split.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help=(
            "Optional fraction of subjects assigned to validation split. "
            "If 0, only train/test splits are created."
        ),
    )

    parser.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Minimum number of sessions required per subject.",
    )

    parser.add_argument(
        "--drop-missing-audio",
        action="store_true",
        help="Drop rows whose audio files do not exist.",
    )

    parser.add_argument(
        "--drop-missing-label",
        action="store_true",
        help="Drop rows with missing labels.",
    )

    parser.add_argument(
        "--drop-missing-transcript",
        action="store_true",
        help="Drop rows with missing transcripts.",
    )

    parser.add_argument(
        "--sort-by-subject-time",
        action="store_true",
        help="Sort cleaned metadata by subject_id and timestamp.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject-level split.",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path to save preparation summary JSON.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Validation and normalization helpers
# ---------------------------------------------------------------------


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Input metadata is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def resolve_audio_path(
    audio_path: Any,
    root_dir: Optional[Path],
) -> str:
    path = Path(str(audio_path))

    if path.is_absolute():
        return str(path)

    if root_dir is not None:
        return str(root_dir / path)

    return str(path)


def audio_exists(
    audio_path: Any,
    root_dir: Optional[Path],
) -> bool:
    resolved = Path(resolve_audio_path(audio_path, root_dir))
    return resolved.exists()


def parse_timestamp(value: Any) -> float:
    """
    Parse timestamp into a numeric float.

    Supports:
    - numeric timestamps;
    - date strings;
    - datetime strings.

    For date/datetime strings, returns Unix timestamp in seconds.
    """

    if pd.isna(value):
        raise ValueError("Missing timestamp.")

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()

    try:
        return float(value_str)
    except ValueError:
        pass

    parsed = pd.to_datetime(value_str, errors="coerce")

    if pd.isna(parsed):
        raise ValueError(f"Could not parse timestamp: {value}")

    return float(parsed.timestamp())


def normalize_labels(
    series: pd.Series,
    mode: str,
    pre_values: Optional[Sequence[str]] = None,
    post_values: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Normalize labels into [0, 1].

    Modes
    -----
    none:
        Use labels as provided, but convert to float.

    minmax:
        Min-max normalize labels.

    alstdi:
        ALSFRS-R speech subscore y is mapped to:
            normalized = (4 - y) / 4
        so larger values indicate greater impairment.

    parkceleb:
        Pre-diagnosis labels are mapped to 0 and post-diagnosis labels to 1.
    """

    if mode == "none":
        return series.astype(float)

    if mode == "minmax":
        values = series.astype(float)
        min_value = values.min()
        max_value = values.max()

        if max_value == min_value:
            return pd.Series([0.0] * len(values), index=values.index)

        return (values - min_value) / (max_value - min_value)

    if mode == "alstdi":
        values = series.astype(float)
        normalized = (4.0 - values) / 4.0
        return normalized.clip(0.0, 1.0)

    if mode == "parkceleb":
        pre_values = set(str(v) for v in (pre_values or []))
        post_values = set(str(v) for v in (post_values or []))

        def convert(value: Any) -> float:
            value_str = str(value).strip()

            if value_str in pre_values:
                return 0.0

            if value_str in post_values:
                return 1.0

            try:
                numeric_value = float(value_str)
                if numeric_value <= 0:
                    return 0.0
                return 1.0
            except ValueError as exc:
                raise ValueError(
                    f"Cannot convert ParkCeleb label value: {value}"
                ) from exc

        return series.apply(convert)

    raise ValueError(f"Unknown label normalization mode: {mode}")


def filter_min_sessions(
    df: pd.DataFrame,
    subject_column: str,
    min_sessions: int,
) -> pd.DataFrame:
    counts = df.groupby(subject_column).size()
    keep_subjects = counts[counts >= min_sessions].index
    return df[df[subject_column].isin(keep_subjects)].copy()


def create_subject_split(
    subjects: Sequence[str],
    test_size: float = 0.2,
    val_size: float = 0.0,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create subject-level train/val/test split.

    test_size and val_size are fractions of all subjects.
    """

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be between 0 and 1.")

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    subjects = list(sorted(str(s) for s in subjects))

    rng = random.Random(seed)
    rng.shuffle(subjects)

    num_subjects = len(subjects)

    num_test = max(1, int(round(num_subjects * test_size)))
    num_val = int(round(num_subjects * val_size)) if val_size > 0 else 0

    test_subjects = subjects[:num_test]
    val_subjects = subjects[num_test : num_test + num_val]
    train_subjects = subjects[num_test + num_val :]

    split = {
        "train": sorted(train_subjects),
        "test": sorted(test_subjects),
    }

    if val_size > 0:
        split["val"] = sorted(val_subjects)

    return split


def summarize_metadata(
    df: pd.DataFrame,
    split: Dict[str, List[str]],
    subject_column: str,
    label_column: str,
    audio_missing_count: int,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Create a compact preparation summary.
    """

    subject_counts = {
        split_name: len(subject_ids)
        for split_name, subject_ids in split.items()
    }

    row_counts = {}
    for split_name, subject_ids in split.items():
        subject_set = set(subject_ids)
        row_counts[split_name] = int(
            df[df[subject_column].isin(subject_set)].shape[0]
        )

    labels = df[label_column].astype(float)

    return {
        "dataset_name": dataset_name,
        "num_rows": int(len(df)),
        "num_subjects": int(df[subject_column].nunique()),
        "num_missing_audio": int(audio_missing_count),
        "label_min": float(labels.min()) if len(labels) else None,
        "label_max": float(labels.max()) if len(labels) else None,
        "label_mean": float(labels.mean()) if len(labels) else None,
        "subject_counts": subject_counts,
        "row_counts": row_counts,
    }


# ---------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------


def prepare_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)

    input_metadata = Path(args.input_metadata)
    output_metadata = Path(args.output_metadata)
    output_split = Path(args.output_split)
    root_dir = Path(args.root_dir) if args.root_dir is not None else None

    if not input_metadata.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_metadata}")

    df = pd.read_csv(input_metadata)

    required_columns = [
        args.subject_column,
        args.audio_column,
        args.transcript_column,
        args.time_column,
        args.label_column,
    ]

    validate_required_columns(df, required_columns)

    original_num_rows = len(df)

    # Basic cleanup.
    df = df.copy()
    df[args.subject_column] = df[args.subject_column].astype(str)
    df[args.audio_column] = df[args.audio_column].astype(str)

    if args.drop_missing_transcript:
        df = df.dropna(subset=[args.transcript_column])
    else:
        df[args.transcript_column] = df[args.transcript_column].fillna("")

    if args.drop_missing_label:
        df = df.dropna(subset=[args.label_column])

    # Parse timestamps.
    df[args.time_column] = df[args.time_column].apply(parse_timestamp)

    # Normalize labels.
    df[args.label_column] = normalize_labels(
        df[args.label_column],
        mode=args.label_mode,
        pre_values=args.pre_diagnosis_values,
        post_values=args.post_diagnosis_values,
    )

    # Resolve audio paths for output metadata.
    df[args.audio_column] = df[args.audio_column].apply(
        lambda path: resolve_audio_path(path, root_dir)
    )

    exists_mask = df[args.audio_column].apply(lambda path: Path(path).exists())
    missing_audio_count = int((~exists_mask).sum())

    if args.drop_missing_audio:
        df = df[exists_mask].copy()

    # Remove subjects with too few sessions.
    df = filter_min_sessions(
        df,
        subject_column=args.subject_column,
        min_sessions=args.min_sessions,
    )

    if len(df) == 0:
        raise ValueError(
            "No rows remain after filtering. "
            "Check min_sessions, label availability, and audio paths."
        )

    if args.sort_by_subject_time:
        df = df.sort_values([args.subject_column, args.time_column])

    # Keep the standardized columns first, but preserve extra columns.
    standard_columns = [
        args.subject_column,
        args.audio_column,
        args.transcript_column,
        args.time_column,
        args.label_column,
    ]

    extra_columns = [col for col in df.columns if col not in standard_columns]
    df = df[standard_columns + extra_columns]

    # Rename to standard names expected by configs.
    rename_map = {
        args.subject_column: "subject_id",
        args.audio_column: "audio_path",
        args.transcript_column: "transcript",
        args.time_column: "timestamp",
        args.label_column: "label",
    }

    df = df.rename(columns=rename_map)

    subjects = sorted(df["subject_id"].unique().tolist())

    split = create_subject_split(
        subjects=subjects,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    ensure_dir(output_metadata.parent)
    ensure_dir(output_split.parent)

    df.to_csv(output_metadata, index=False)

    with output_split.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    summary = summarize_metadata(
        df=df,
        split=split,
        subject_column="subject_id",
        label_column="label",
        audio_missing_count=missing_audio_count,
        dataset_name=args.dataset_name,
    )

    summary["input_metadata"] = str(input_metadata)
    summary["output_metadata"] = str(output_metadata)
    summary["output_split"] = str(output_split)
    summary["original_num_rows"] = int(original_num_rows)
    summary["final_num_rows"] = int(len(df))
    summary["min_sessions"] = int(args.min_sessions)
    summary["label_mode"] = args.label_mode

    if args.summary_path is not None:
        save_json(summary, args.summary_path)

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    summary = prepare_metadata(args)

    print("\nData preparation complete.")
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Rows: {summary['final_num_rows']}")
    print(f"Subjects: {summary['num_subjects']}")
    print(f"Missing audio files: {summary['num_missing_audio']}")
    print(f"Saved metadata to: {summary['output_metadata']}")
    print(f"Saved split to: {summary['output_split']}")

    print("\nSplit summary:")
    for split_name, count in summary["subject_counts"].items():
        rows = summary["row_counts"][split_name]
        print(f"  {split_name}: {count} subjects, {rows} rows")


if __name__ == "__main__":
    main()