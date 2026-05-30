"""
This module handles:
1. Loading metadata CSV files.
2. Loading subject-level train/test splits.
3. Grouping recordings into longitudinal subject trajectories.
4. Creating training samples where one session is the target and
   the remaining sessions are used as longitudinal context.
5. Creating inference intervals for intermediate synthetic generation.

Expected metadata columns:
    subject_id,audio_path,transcript,timestamp,label

Expected split JSON format:
    {
        "train": ["subject_001", "subject_002"],
        "test": ["subject_101", "subject_102"]
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RecordingSession:
    """
    A single speech recording session.

    Attributes
    ----------
    subject_id:
        Unique subject identifier.
    audio_path:
        Path to the speech recording.
    transcript:
        Text transcript or prompt associated with the recording.
    timestamp:
        Numeric timestamp used for longitudinal ordering.
        Internally, timestamps are converted to float values.
    label:
        Normalized clinical progression label in [0, 1].
    metadata:
        Optional extra fields from the metadata CSV.
    """

    subject_id: str
    audio_path: str
    transcript: str
    timestamp: float
    label: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class LongitudinalSubject:
    """
    A subject-level longitudinal trajectory.

    Attributes
    ----------
    subject_id:
        Unique subject identifier.
    sessions:
        Ordered list of recording sessions for this subject.
    """

    subject_id: str
    sessions: List[RecordingSession]


@dataclass(frozen=True)
class InferenceQuery:
    """
    A synthetic generation query between two neighboring real sessions.

    Attributes
    ----------
    subject_id:
        Unique subject identifier.
    query_timestamp:
        Intermediate timestamp where synthetic speech should be generated.
    soft_label:
        Interpolated progression label at the query timestamp.
    previous_session:
        Previous real recording session.
    next_session:
        Next real recording session.
    context_sessions:
        All observed real sessions for the subject.
    """

    subject_id: str
    query_timestamp: float
    soft_label: float
    previous_session: RecordingSession
    next_session: RecordingSession
    context_sessions: List[RecordingSession]


def load_split(split_path: PathLike, split_name: str) -> List[str]:
    """
    Load subject IDs for a given split.

    Parameters
    ----------
    split_path:
        Path to split JSON file.
    split_name:
        Split name, usually "train", "val", or "test".

    Returns
    -------
    List[str]
        Subject IDs belonging to the selected split.
    """

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split_name not in split_data:
        available = list(split_data.keys())
        raise KeyError(
            f"Split '{split_name}' not found in {split_path}. "
            f"Available splits: {available}"
        )

    return [str(subject_id) for subject_id in split_data[split_name]]


def _convert_timestamp(value: Any) -> float:
    """
    Convert a timestamp value to a numeric float.

    This function supports:
    - numeric values, such as days from baseline;
    - date strings, such as "2018-01-01";
    - datetime-like strings.

    For date strings, the returned value is the Unix timestamp in seconds.
    Later preprocessing can convert this into days from baseline if needed.
    """

    if pd.isna(value):
        raise ValueError("Timestamp value is missing.")

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()

    try:
        return float(value_str)
    except ValueError:
        pass

    parsed = pd.to_datetime(value_str, errors="coerce")

    if pd.isna(parsed):
        raise ValueError(f"Unable to parse timestamp value: {value}")

    return float(parsed.timestamp())


def _safe_float(value: Any, field_name: str) -> float:
    """
    Convert value to float with a useful error message.
    """

    if pd.isna(value):
        raise ValueError(f"Missing value for required field '{field_name}'.")

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Could not convert field '{field_name}' to float. "
            f"Received value: {value}"
        ) from exc


def load_metadata(
    metadata_path: PathLike,
    root_dir: Optional[PathLike] = None,
    subject_column: str = "subject_id",
    audio_column: str = "audio_path",
    transcript_column: str = "transcript",
    time_column: str = "timestamp",
    label_column: str = "label",
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load and validate metadata CSV.

    Parameters
    ----------
    metadata_path:
        Path to metadata CSV.
    root_dir:
        Optional dataset root directory. If provided, relative audio paths are
        resolved against this directory.
    subject_column, audio_column, transcript_column, time_column, label_column:
        Metadata column names.
    required_columns:
        Optional additional required columns.

    Returns
    -------
    pd.DataFrame
        Validated metadata dataframe.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    base_required = [
        subject_column,
        audio_column,
        transcript_column,
        time_column,
        label_column,
    ]

    if required_columns is not None:
        base_required.extend(required_columns)

    missing = [col for col in base_required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[subject_column] = df[subject_column].astype(str)
    df[transcript_column] = df[transcript_column].fillna("").astype(str)
    df[label_column] = df[label_column].apply(
        lambda x: _safe_float(x, label_column)
    )
    df[time_column] = df[time_column].apply(_convert_timestamp)

    if root_dir is not None:
        root_dir = Path(root_dir)

        def resolve_audio_path(path_value: Any) -> str:
            path = Path(str(path_value))
            if path.is_absolute():
                return str(path)
            return str(root_dir / path)

        df[audio_column] = df[audio_column].apply(resolve_audio_path)
    else:
        df[audio_column] = df[audio_column].astype(str)

    return df


def build_subject_trajectories(
    df: pd.DataFrame,
    subject_ids: Optional[Sequence[str]] = None,
    subject_column: str = "subject_id",
    audio_column: str = "audio_path",
    transcript_column: str = "transcript",
    time_column: str = "timestamp",
    label_column: str = "label",
    min_sessions: int = 2,
) -> List[LongitudinalSubject]:
    """
    Convert metadata dataframe into ordered subject-level trajectories.

    Parameters
    ----------
    df:
        Metadata dataframe.
    subject_ids:
        Optional list of subject IDs to keep.
    min_sessions:
        Minimum number of sessions required for a subject to be included.

    Returns
    -------
    List[LongitudinalSubject]
        Subject trajectories sorted by timestamp.
    """

    if subject_ids is not None:
        subject_ids = set(str(x) for x in subject_ids)
        df = df[df[subject_column].astype(str).isin(subject_ids)]

    subjects: List[LongitudinalSubject] = []

    for subject_id, group in df.groupby(subject_column):
        group = group.sort_values(time_column)

        if len(group) < min_sessions:
            continue

        sessions: List[RecordingSession] = []

        for _, row in group.iterrows():
            reserved = {
                subject_column,
                audio_column,
                transcript_column,
                time_column,
                label_column,
            }

            extra_metadata = {
                col: row[col]
                for col in df.columns
                if col not in reserved
            }

            session = RecordingSession(
                subject_id=str(row[subject_column]),
                audio_path=str(row[audio_column]),
                transcript=str(row[transcript_column]),
                timestamp=float(row[time_column]),
                label=float(row[label_column]),
                metadata=extra_metadata,
            )
            sessions.append(session)

        subjects.append(
            LongitudinalSubject(
                subject_id=str(subject_id),
                sessions=sessions,
            )
        )

    subjects.sort(key=lambda item: item.subject_id)
    return subjects


class LongitudinalSpeechDataset(Dataset):
    """
    PyTorch dataset for L-Flow training.

    Each item contains:
    - one target session;
    - all other sessions from the same subject as context;
    - the nearest reference session in time.

    This matches the training setting in L-Flow where one recording q is used
    as the target and H_i \\ {q} is used as longitudinal context.

    Notes
    -----
    This dataset does not load waveform or mel-spectrogram features by default.
    Instead, it returns metadata objects and paths. Actual audio/mel loading
    should be handled by preprocessing transforms to keep this dataset flexible.
    """

    def __init__(
        self,
        metadata_path: PathLike,
        split_path: Optional[PathLike] = None,
        split: str = "train",
        root_dir: Optional[PathLike] = None,
        subject_column: str = "subject_id",
        audio_column: str = "audio_path",
        transcript_column: str = "transcript",
        time_column: str = "timestamp",
        label_column: str = "label",
        min_sessions: int = 2,
        target_sampling: str = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        metadata_path:
            Path to metadata CSV.
        split_path:
            Optional path to subject-level split JSON.
        split:
            Split to load from split JSON.
        root_dir:
            Optional dataset root directory for resolving relative audio paths.
        min_sessions:
            Minimum number of sessions required for a subject to be included.
        target_sampling:
            "all" creates one training item for each session of each subject.
            "random" creates one item per subject and randomly samples the target
            session whenever __getitem__ is called.
        transform:
            Optional transform applied to each returned item.
        """

        if target_sampling not in {"all", "random"}:
            raise ValueError(
                f"target_sampling must be 'all' or 'random', got {target_sampling}"
            )

        self.metadata_path = Path(metadata_path)
        self.split_path = Path(split_path) if split_path is not None else None
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None

        self.subject_column = subject_column
        self.audio_column = audio_column
        self.transcript_column = transcript_column
        self.time_column = time_column
        self.label_column = label_column
        self.min_sessions = min_sessions
        self.target_sampling = target_sampling
        self.transform = transform

        df = load_metadata(
            metadata_path=self.metadata_path,
            root_dir=self.root_dir,
            subject_column=subject_column,
            audio_column=audio_column,
            transcript_column=transcript_column,
            time_column=time_column,
            label_column=label_column,
        )

        subject_ids: Optional[List[str]] = None
        if self.split_path is not None:
            subject_ids = load_split(self.split_path, split)

        self.subjects = build_subject_trajectories(
            df=df,
            subject_ids=subject_ids,
            subject_column=subject_column,
            audio_column=audio_column,
            transcript_column=transcript_column,
            time_column=time_column,
            label_column=label_column,
            min_sessions=min_sessions,
        )

        if len(self.subjects) == 0:
            raise ValueError(
                "No valid subjects found. Check metadata, split file, "
                f"and min_sessions={min_sessions}."
            )

        self.index: List[Tuple[int, Optional[int]]] = []

        if self.target_sampling == "all":
            for subject_idx, subject in enumerate(self.subjects):
                for target_idx in range(len(subject.sessions)):
                    self.index.append((subject_idx, target_idx))
        else:
            for subject_idx, _ in enumerate(self.subjects):
                self.index.append((subject_idx, None))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject_idx, target_idx = self.index[idx]
        subject = self.subjects[subject_idx]

        if target_idx is None:
            target_idx = random.randrange(len(subject.sessions))

        target_session = subject.sessions[target_idx]
        context_sessions = [
            session
            for i, session in enumerate(subject.sessions)
            if i != target_idx
        ]

        reference_session = select_nearest_reference_session(
            target_session=target_session,
            candidate_sessions=context_sessions,
            tie_break="earlier",
        )

        item: Dict[str, Any] = {
            "subject_id": subject.subject_id,
            "target": session_to_dict(target_session),
            "context": [session_to_dict(s) for s in context_sessions],
            "reference": session_to_dict(reference_session),
            "num_context_sessions": len(context_sessions),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


class LongitudinalInferenceDataset(Dataset):
    """
    Dataset for generating intermediate synthetic samples.

    For each subject trajectory, this dataset creates query points between
    neighboring sessions. The number of query points per interval follows:

        M(a, b) = min(Mmax, max(0, floor((t_b - t_a) / h) - 1))

    The soft progression label is linearly interpolated between the previous
    and next real sessions.
    """

    def __init__(
        self,
        metadata_path: PathLike,
        split_path: Optional[PathLike] = None,
        split: str = "train",
        root_dir: Optional[PathLike] = None,
        subject_column: str = "subject_id",
        audio_column: str = "audio_path",
        transcript_column: str = "transcript",
        time_column: str = "timestamp",
        label_column: str = "label",
        min_sessions: int = 2,
        temporal_resolution: float = 30.0,
        max_samples_per_interval: int = 2,
        time_unit: str = "days",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        temporal_resolution:
            Desired temporal resolution h.

            If time_unit="days" and timestamps are date strings, timestamps are
            internally converted to seconds, and h is converted to seconds.

            If your metadata already stores days-from-baseline as numeric values,
            set time_unit="raw".
        max_samples_per_interval:
            Maximum number of synthetic samples per neighboring interval.
        time_unit:
            "days" or "raw".
        transform:
            Optional transform applied to each returned item.
        """

        if temporal_resolution <= 0:
            raise ValueError("temporal_resolution must be positive.")

        if max_samples_per_interval < 0:
            raise ValueError("max_samples_per_interval must be non-negative.")

        if time_unit not in {"days", "raw"}:
            raise ValueError("time_unit must be either 'days' or 'raw'.")

        self.metadata_path = Path(metadata_path)
        self.split_path = Path(split_path) if split_path is not None else None
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None

        self.subject_column = subject_column
        self.audio_column = audio_column
        self.transcript_column = transcript_column
        self.time_column = time_column
        self.label_column = label_column

        self.min_sessions = min_sessions
        self.temporal_resolution = temporal_resolution
        self.max_samples_per_interval = max_samples_per_interval
        self.time_unit = time_unit
        self.transform = transform

        df = load_metadata(
            metadata_path=self.metadata_path,
            root_dir=self.root_dir,
            subject_column=subject_column,
            audio_column=audio_column,
            transcript_column=transcript_column,
            time_column=time_column,
            label_column=label_column,
        )

        subject_ids: Optional[List[str]] = None
        if self.split_path is not None:
            subject_ids = load_split(self.split_path, split)

        self.subjects = build_subject_trajectories(
            df=df,
            subject_ids=subject_ids,
            subject_column=subject_column,
            audio_column=audio_column,
            transcript_column=transcript_column,
            time_column=time_column,
            label_column=label_column,
            min_sessions=min_sessions,
        )

        resolution = self._get_internal_resolution()
        self.queries = build_interpolation_queries(
            subjects=self.subjects,
            temporal_resolution=resolution,
            max_samples_per_interval=max_samples_per_interval,
        )

    def _get_internal_resolution(self) -> float:
        if self.time_unit == "days":
            return float(self.temporal_resolution) * 24.0 * 60.0 * 60.0
        return float(self.temporal_resolution)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query = self.queries[idx]

        reference_session = select_nearest_reference_session(
            target_timestamp=query.query_timestamp,
            candidate_sessions=query.context_sessions,
            tie_break="earlier",
        )

        item: Dict[str, Any] = {
            "subject_id": query.subject_id,
            "query_timestamp": query.query_timestamp,
            "soft_label": query.soft_label,
            "previous_session": session_to_dict(query.previous_session),
            "next_session": session_to_dict(query.next_session),
            "context": [session_to_dict(s) for s in query.context_sessions],
            "reference": session_to_dict(reference_session),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


def build_interpolation_queries(
    subjects: Sequence[LongitudinalSubject],
    temporal_resolution: float,
    max_samples_per_interval: int,
) -> List[InferenceQuery]:
    """
    Build intermediate query points between neighboring sessions.

    Parameters
    ----------
    subjects:
        Subject trajectories.
    temporal_resolution:
        Desired temporal resolution h, in the same unit as session timestamps.
    max_samples_per_interval:
        Maximum number of synthetic samples per interval.

    Returns
    -------
    List[InferenceQuery]
        Query points for synthetic generation.
    """

    queries: List[InferenceQuery] = []

    for subject in subjects:
        sessions = subject.sessions

        if len(sessions) < 2:
            continue

        for left, right in zip(sessions[:-1], sessions[1:]):
            gap = right.timestamp - left.timestamp

            if gap <= 0:
                continue

            num_samples = int(gap // temporal_resolution) - 1
            num_samples = max(0, num_samples)
            num_samples = min(max_samples_per_interval, num_samples)

            if num_samples <= 0:
                continue

            # Uniformly place query points inside the interval.
            # Example: M=2 gives alpha = 1/3 and 2/3.
            for sample_idx in range(1, num_samples + 1):
                alpha = sample_idx / (num_samples + 1)
                query_timestamp = (1.0 - alpha) * left.timestamp + alpha * right.timestamp
                soft_label = (1.0 - alpha) * left.label + alpha * right.label

                queries.append(
                    InferenceQuery(
                        subject_id=subject.subject_id,
                        query_timestamp=float(query_timestamp),
                        soft_label=float(soft_label),
                        previous_session=left,
                        next_session=right,
                        context_sessions=list(sessions),
                    )
                )

    return queries


def select_nearest_reference_session(
    candidate_sessions: Sequence[RecordingSession],
    target_session: Optional[RecordingSession] = None,
    target_timestamp: Optional[float] = None,
    tie_break: str = "earlier",
) -> RecordingSession:
    """
    Select the reference speech by temporal proximity.

    Parameters
    ----------
    candidate_sessions:
        Candidate real sessions.
    target_session:
        Target session. Used during training.
    target_timestamp:
        Query timestamp. Used during inference.
    tie_break:
        "earlier" or "later" when two sessions are equally close.

    Returns
    -------
    RecordingSession
        Nearest reference session.
    """

    if tie_break not in {"earlier", "later"}:
        raise ValueError("tie_break must be either 'earlier' or 'later'.")

    if len(candidate_sessions) == 0:
        raise ValueError("candidate_sessions must not be empty.")

    if target_session is None and target_timestamp is None:
        raise ValueError("Either target_session or target_timestamp must be provided.")

    if target_timestamp is None:
        assert target_session is not None
        target_timestamp = target_session.timestamp

    def sort_key(session: RecordingSession) -> Tuple[float, float]:
        distance = abs(session.timestamp - float(target_timestamp))

        if tie_break == "earlier":
            # Earlier timestamps should come first in a tie.
            tie_value = session.timestamp
        else:
            # Later timestamps should come first in a tie.
            tie_value = -session.timestamp

        return distance, tie_value

    return sorted(candidate_sessions, key=sort_key)[0]


def session_to_dict(session: RecordingSession) -> Dict[str, Any]:
    """
    Convert RecordingSession dataclass to dictionary.
    """

    return {
        "subject_id": session.subject_id,
        "audio_path": session.audio_path,
        "transcript": session.transcript,
        "timestamp": session.timestamp,
        "label": session.label,
        "metadata": dict(session.metadata),
    }


def longitudinal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate function for variable-length longitudinal contexts.

    This function keeps nested metadata as Python lists because each subject
    can have a different number of context sessions. Audio/mel tensors can be
    added later by preprocessing transforms and padded in a more specialized
    collate function if needed.
    """

    return {
        "subject_id": [item["subject_id"] for item in batch],
        "target": [item.get("target") for item in batch],
        "context": [item["context"] for item in batch],
        "reference": [item["reference"] for item in batch],
        "num_context_sessions": [
            item.get("num_context_sessions", len(item["context"]))
            for item in batch
        ],
    }


def inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for inference query batches.
    """

    return {
        "subject_id": [item["subject_id"] for item in batch],
        "query_timestamp": torch.tensor(
            [item["query_timestamp"] for item in batch],
            dtype=torch.float32,
        ),
        "soft_label": torch.tensor(
            [item["soft_label"] for item in batch],
            dtype=torch.float32,
        ),
        "previous_session": [item["previous_session"] for item in batch],
        "next_session": [item["next_session"] for item in batch],
        "context": [item["context"] for item in batch],
        "reference": [item["reference"] for item in batch],
    }


def create_dataset_from_config(
    config: Dict[str, Any],
    split: str = "train",
    mode: str = "train",
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dataset:
    """
    Factory function for creating datasets from a config dictionary.

    Parameters
    ----------
    config:
        Loaded YAML config.
    split:
        Dataset split name.
    mode:
        "train" for LongitudinalSpeechDataset,
        "inference" for LongitudinalInferenceDataset.
    transform:
        Optional transform.

    Returns
    -------
    Dataset
        A PyTorch Dataset instance.
    """

    if mode not in {"train", "inference"}:
        raise ValueError("mode must be either 'train' or 'inference'.")

    dataset_cfg = config.get("dataset", {})

    common_kwargs = {
        "metadata_path": dataset_cfg["metadata_path"],
        "split_path": dataset_cfg.get("split_path"),
        "split": split,
        "root_dir": dataset_cfg.get("root_dir"),
        "subject_column": dataset_cfg.get("subject_column", "subject_id"),
        "audio_column": dataset_cfg.get("audio_column", "audio_path"),
        "transcript_column": dataset_cfg.get("transcript_column", "transcript"),
        "time_column": dataset_cfg.get("time_column", "timestamp"),
        "label_column": dataset_cfg.get("label_column", "label"),
        "transform": transform,
    }

    if mode == "train":
        return LongitudinalSpeechDataset(
            **common_kwargs,
            min_sessions=dataset_cfg.get("min_sessions", 2),
            target_sampling=dataset_cfg.get("target_sampling", "all"),
        )

    generation_cfg = config.get("generation", {})

    return LongitudinalInferenceDataset(
        **common_kwargs,
        min_sessions=dataset_cfg.get("min_sessions", 2),
        temporal_resolution=generation_cfg.get("temporal_resolution_days", 30),
        max_samples_per_interval=generation_cfg.get("max_samples_per_interval", 2),
        time_unit=generation_cfg.get("time_unit", "days"),
    )