"""

This module evaluates whether generated synthetic samples preserve the expected
longitudinal progression ordering within each subject trajectory.

Metrics:
    1. Pairwise Accuracy
    2. Kendall's tau
    3. Spearman's rho

Typical usage
-------------
The input should be a CSV containing generated synthetic samples and predicted
severity scores, for example:

    subject_id,timestamp,label,predicted_score,audio_path,method

Where:
    subject_id:
        Subject identifier.

    timestamp:
        Query timestamp for the generated sample.

    label:
        Soft progression label assigned by interpolation.

    predicted_score:
        Severity score predicted by an external regressor trained on real data.

This module does not require the audio itself. It evaluates the ordering of
predicted severity scores relative to the expected ordering from timestamps
or soft labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from lflow.utils import ensure_dir, save_csv_rows, save_json, setup_logger


# ---------------------------------------------------------------------
# Basic metric utilities
# ---------------------------------------------------------------------


def safe_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """
    Compute Spearman correlation safely.

    Returns 0.0 if correlation is undefined.
    """

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) < 2:
        return 0.0

    if len(np.unique(x_arr)) <= 1 or len(np.unique(y_arr)) <= 1:
        return 0.0

    result = spearmanr(x_arr, y_arr)
    value = float(result.correlation)

    if np.isnan(value):
        return 0.0

    return value


def safe_kendall(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """
    Compute Kendall's tau safely.

    Returns 0.0 if correlation is undefined.
    """

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) < 2:
        return 0.0

    if len(np.unique(x_arr)) <= 1 or len(np.unique(y_arr)) <= 1:
        return 0.0

    result = kendalltau(x_arr, y_arr)
    value = float(result.correlation)

    if np.isnan(value):
        return 0.0

    return value


def pairwise_accuracy(
    expected_order_values: Sequence[float],
    predicted_scores: Sequence[float],
    min_gap: float = 1e-8,
) -> float:
    """
    Compute pairwise ordering accuracy.

    Parameters
    ----------
    expected_order_values:
        Values defining expected ordering. Usually timestamps or soft labels.
        Larger values should correspond to more advanced progression.

    predicted_scores:
        Predicted severity/progression scores.

    min_gap:
        Minimum difference required to consider a pair ordered.

    Returns
    -------
    float
        Fraction of correctly ordered pairs.
    """

    expected = np.asarray(expected_order_values, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_scores, dtype=float).reshape(-1)

    if len(expected) != len(predicted):
        raise ValueError(
            f"Expected and predicted arrays must have the same length. "
            f"Got {len(expected)} and {len(predicted)}."
        )

    n = len(expected)

    if n < 2:
        return 0.0

    total = 0
    correct = 0

    for i in range(n):
        for j in range(i + 1, n):
            expected_diff = expected[j] - expected[i]
            predicted_diff = predicted[j] - predicted[i]

            if abs(expected_diff) <= min_gap:
                continue

            total += 1

            if np.sign(expected_diff) == np.sign(predicted_diff):
                correct += 1

    if total == 0:
        return 0.0

    return float(correct / total)


# ---------------------------------------------------------------------
# Subject-level evaluation
# ---------------------------------------------------------------------


def evaluate_subject_progression(
    df_subject: pd.DataFrame,
    order_column: str = "timestamp",
    score_column: str = "predicted_score",
    min_pairs: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate progression consistency for one subject.

    Parameters
    ----------
    df_subject:
        Dataframe containing generated samples from one subject.

    order_column:
        Column defining expected ordering. Usually "timestamp" or "label".

    score_column:
        Column containing predicted severity scores.

    min_pairs:
        Minimum number of ordered pairs required for metric computation.

    Returns
    -------
    dict
        Subject-level metrics.
    """

    df_subject = df_subject.sort_values(order_column).reset_index(drop=True)

    expected = df_subject[order_column].astype(float).to_numpy()
    predicted = df_subject[score_column].astype(float).to_numpy()

    n = len(df_subject)
    num_pairs = n * (n - 1) // 2

    if n < 2 or num_pairs < min_pairs:
        return {
            "subject_id": df_subject["subject_id"].iloc[0],
            "num_samples": int(n),
            "num_pairs": int(num_pairs),
            "pairwise_accuracy": 0.0,
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "valid": False,
        }

    return {
        "subject_id": df_subject["subject_id"].iloc[0],
        "num_samples": int(n),
        "num_pairs": int(num_pairs),
        "pairwise_accuracy": pairwise_accuracy(expected, predicted),
        "kendall_tau": safe_kendall(expected, predicted),
        "spearman_rho": safe_spearman(expected, predicted),
        "valid": True,
    }


def aggregate_subject_metrics(
    subject_rows: Sequence[Dict[str, Any]],
    weighted: bool = True,
) -> Dict[str, float]:
    """
    Aggregate subject-level progression metrics.

    Parameters
    ----------
    subject_rows:
        List of subject-level metric dictionaries.

    weighted:
        If True, weight each subject by number of valid pairs.

    Returns
    -------
    dict
        Aggregated metrics.
    """

    valid_rows = [row for row in subject_rows if row.get("valid", False)]

    if len(valid_rows) == 0:
        return {
            "pairwise_accuracy": 0.0,
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
        }

    metric_names = [
        "pairwise_accuracy",
        "kendall_tau",
        "spearman_rho",
    ]

    output: Dict[str, float] = {}

    if weighted:
        weights = np.asarray(
            [max(1, int(row.get("num_pairs", 1))) for row in valid_rows],
            dtype=float,
        )
        weights = weights / weights.sum()

        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in valid_rows])
            output[metric] = float(np.sum(values * weights))

    else:
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in valid_rows])
            output[metric] = float(np.mean(values))

    return output


# ---------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------


def validate_input_dataframe(
    df: pd.DataFrame,
    subject_column: str,
    order_column: str,
    score_column: str,
) -> None:
    """
    Validate required columns.
    """

    required = [
        subject_column,
        order_column,
        score_column,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def evaluate_progression_consistency(
    input_csv: str,
    output_dir: str,
    subject_column: str = "subject_id",
    order_column: str = "timestamp",
    score_column: str = "predicted_score",
    method_column: Optional[str] = "method",
    weighted: bool = True,
    min_pairs: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate progression consistency for generated samples.

    Parameters
    ----------
    input_csv:
        CSV file containing synthetic samples and predicted severity scores.

    output_dir:
        Directory to save subject-level and summary metrics.

    subject_column:
        Column containing subject IDs.

    order_column:
        Column defining expected progression ordering.
        Common choices:
            - "timestamp"
            - "label"

    score_column:
        Column containing predicted severity scores.

    method_column:
        Optional method column. If present, metrics are computed separately
        for each method.

    weighted:
        Whether to aggregate subject metrics weighted by number of pairs.

    min_pairs:
        Minimum number of pairs required to treat a subject as valid.

    Returns
    -------
    dict
        Summary metrics.
    """

    input_path = Path(input_csv)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    ensure_dir(output_path)

    logger = setup_logger(
        name="evaluation.progression_consistency",
        log_file=output_path / "progression_consistency.log",
    )

    logger.info(f"Loading input CSV: {input_path}")

    df = pd.read_csv(input_path)

    # Standardize subject column name internally.
    if subject_column != "subject_id":
        df = df.rename(columns={subject_column: "subject_id"})
        subject_column = "subject_id"

    validate_input_dataframe(
        df=df,
        subject_column=subject_column,
        order_column=order_column,
        score_column=score_column,
    )

    df = df.copy()
    df[subject_column] = df[subject_column].astype(str)
    df[order_column] = df[order_column].astype(float)
    df[score_column] = df[score_column].astype(float)

    if method_column is not None and method_column in df.columns:
        methods = sorted(df[method_column].dropna().astype(str).unique().tolist())
    else:
        method_column = None
        methods = ["all"]
        df["_method"] = "all"
        method_column = "_method"

    all_subject_rows: List[Dict[str, Any]] = []
    summary_by_method: Dict[str, Any] = {}

    for method in methods:
        df_method = df[df[method_column].astype(str) == method].copy()

        subject_rows: List[Dict[str, Any]] = []

        for subject_id, df_subject in df_method.groupby(subject_column):
            row = evaluate_subject_progression(
                df_subject=df_subject,
                order_column=order_column,
                score_column=score_column,
                min_pairs=min_pairs,
            )
            row["method"] = method
            subject_rows.append(row)

        aggregate = aggregate_subject_metrics(
            subject_rows=subject_rows,
            weighted=weighted,
        )

        summary_by_method[method] = {
            **aggregate,
            "num_subjects": int(len(subject_rows)),
            "num_valid_subjects": int(sum(row.get("valid", False) for row in subject_rows)),
            "num_samples": int(len(df_method)),
        }

        all_subject_rows.extend(subject_rows)

        logger.info(
            f"{method}: "
            f"Pairwise Acc={aggregate['pairwise_accuracy']:.4f}, "
            f"Kendall={aggregate['kendall_tau']:.4f}, "
            f"Spearman={aggregate['spearman_rho']:.4f}"
        )

    subject_metrics_path = output_path / "subject_progression_metrics.csv"
    summary_path = output_path / "progression_consistency_summary.json"

    save_csv_rows(
        rows=all_subject_rows,
        path=subject_metrics_path,
        fieldnames=[
            "method",
            "subject_id",
            "num_samples",
            "num_pairs",
            "pairwise_accuracy",
            "kendall_tau",
            "spearman_rho",
            "valid",
        ],
    )

    summary = {
        "input_csv": str(input_path),
        "output_dir": str(output_path),
        "order_column": order_column,
        "score_column": score_column,
        "weighted": weighted,
        "min_pairs": min_pairs,
        "methods": summary_by_method,
        "subject_metrics_path": str(subject_metrics_path),
    }

    save_json(summary, summary_path)

    logger.info(f"Saved subject metrics to {subject_metrics_path}")
    logger.info(f"Saved summary to {summary_path}")

    return summary


# ---------------------------------------------------------------------
# Optional helper: merge synthetic metadata with predicted scores
# ---------------------------------------------------------------------


def attach_predicted_scores(
    metadata_csv: str,
    predictions_csv: str,
    output_csv: str,
    metadata_key: str = "audio_path",
    prediction_key: str = "audio_path",
    score_column: str = "predicted_score",
) -> Dict[str, Any]:
    """
    Merge synthetic metadata with predicted severity scores.

    This is useful when predictions are generated by another script.

    Parameters
    ----------
    metadata_csv:
        Synthetic metadata CSV.

    predictions_csv:
        CSV containing predicted scores.

    output_csv:
        Output merged CSV.

    metadata_key:
        Join key in metadata CSV.

    prediction_key:
        Join key in predictions CSV.

    score_column:
        Name of the predicted score column in predictions CSV.

    Returns
    -------
    dict
        Merge summary.
    """

    metadata_path = Path(metadata_csv)
    predictions_path = Path(predictions_csv)
    output_path = Path(output_csv)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    meta_df = pd.read_csv(metadata_path)
    pred_df = pd.read_csv(predictions_path)

    if metadata_key not in meta_df.columns:
        raise ValueError(f"{metadata_key} not found in metadata CSV.")

    if prediction_key not in pred_df.columns:
        raise ValueError(f"{prediction_key} not found in predictions CSV.")

    if score_column not in pred_df.columns:
        raise ValueError(f"{score_column} not found in predictions CSV.")

    merged = meta_df.merge(
        pred_df[[prediction_key, score_column]],
        left_on=metadata_key,
        right_on=prediction_key,
        how="inner",
    )

    ensure_dir(output_path.parent)
    merged.to_csv(output_path, index=False)

    summary = {
        "metadata_rows": int(len(meta_df)),
        "prediction_rows": int(len(pred_df)),
        "merged_rows": int(len(merged)),
        "output_csv": str(output_path),
    }

    save_json(summary, output_path.parent / "attach_scores_summary.json")

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate progression consistency of synthetic speech."
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help=(
            "CSV containing synthetic samples and predicted severity scores. "
            "Must contain subject_id, timestamp or label, and predicted_score."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save progression consistency metrics.",
    )

    parser.add_argument(
        "--subject-column",
        type=str,
        default="subject_id",
        help="Subject ID column.",
    )

    parser.add_argument(
        "--order-column",
        type=str,
        default="timestamp",
        choices=["timestamp", "label"],
        help=(
            "Column defining expected progression ordering. "
            "Use timestamp for temporal consistency or label for severity consistency."
        ),
    )

    parser.add_argument(
        "--score-column",
        type=str,
        default="predicted_score",
        help="Column containing predicted severity scores.",
    )

    parser.add_argument(
        "--method-column",
        type=str,
        default="method",
        help="Optional method column for method-specific metrics.",
    )

    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted mean across subjects.",
    )

    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        help="Minimum number of ordered pairs required per subject.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_progression_consistency(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        subject_column=args.subject_column,
        order_column=args.order_column,
        score_column=args.score_column,
        method_column=args.method_column,
        weighted=not args.unweighted,
        min_pairs=args.min_pairs,
    )


if __name__ == "__main__":
    main()