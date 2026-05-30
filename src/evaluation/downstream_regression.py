"""

Downstream clinical severity regression evaluation for L-Flow.

This module trains downstream regressors on:

    D_aug = D_real ∪ D_syn

and evaluates them on held-out real test subjects.

Expected metadata columns:
    subject_id,audio_path,transcript,timestamp,label

Expected split JSON:
    {
        "train": ["subject_001", "subject_002"],
        "test": ["subject_101", "subject_102"]
    }

Main entry point:
    run_downstream_evaluation(config)
"""

from __future__ import annotations

import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from lflow.utils import (
    ensure_dir,
    get_device_from_config,
    save_csv_rows,
    save_json,
    save_yaml,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------


def load_split(split_path: str) -> Dict[str, List[str]]:
    """
    Load subject-level split JSON.
    """

    path = Path(split_path)

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    if "train" not in split or "test" not in split:
        raise ValueError("Split JSON must contain at least 'train' and 'test' keys.")

    return {
        key: [str(x) for x in values]
        for key, values in split.items()
    }


def load_metadata(path: str) -> pd.DataFrame:
    """
    Load metadata CSV.
    """

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Metadata file not found: {path_obj}")

    df = pd.read_csv(path_obj)

    required = ["subject_id", "audio_path", "label"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata file {path_obj} is missing required columns: {missing}"
        )

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["audio_path"] = df["audio_path"].astype(str)
    df["label"] = df["label"].astype(float)

    if "transcript" not in df.columns:
        df["transcript"] = ""

    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df), dtype=float)

    return df


def filter_by_subjects(
    df: pd.DataFrame,
    subject_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Keep rows whose subject_id belongs to subject_ids.
    """

    subject_set = set(str(x) for x in subject_ids)
    return df[df["subject_id"].astype(str).isin(subject_set)].copy()


def build_train_test_metadata(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train and test metadata.

    Training set:
        real training rows
        optionally synthetic training rows

    Test set:
        real test rows only
    """

    dataset_cfg = config.get("dataset", {})

    real_df = load_metadata(dataset_cfg["real_metadata_path"])
    split = load_split(dataset_cfg["split_path"])

    train_real = filter_by_subjects(real_df, split["train"])
    test_real = filter_by_subjects(real_df, split["test"])

    if len(train_real) == 0:
        raise ValueError("No real training rows found after subject filtering.")

    if len(test_real) == 0:
        raise ValueError("No real test rows found after subject filtering.")

    train_with_synthetic = bool(dataset_cfg.get("train_with_synthetic", True))

    if train_with_synthetic:
        synthetic_path = dataset_cfg.get("synthetic_metadata_path")

        if synthetic_path is None:
            raise ValueError(
                "train_with_synthetic=True but synthetic_metadata_path is missing."
            )

        syn_df = load_metadata(synthetic_path)

        # Synthetic metadata should already be generated only for train subjects,
        # but filter again for safety.
        syn_df = filter_by_subjects(syn_df, split["train"])

        # Remove rows without generated audio.
        syn_df = syn_df[syn_df["audio_path"].notna()].copy()
        syn_df = syn_df[syn_df["audio_path"].astype(str) != "None"].copy()

        train_df = pd.concat([train_real, syn_df], ignore_index=True)
    else:
        train_df = train_real.copy()

    return train_df, test_real


# ---------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    max_seconds: Optional[float] = 30.0,
) -> torch.Tensor:
    """
    Load one waveform and resample to target sample rate.

    Returns
    -------
    waveform:
        Tensor shape [T].
    """

    path = Path(audio_path)

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
        max_samples = int(max_seconds * sample_rate)
        if waveform.numel() > max_samples:
            waveform = waveform[:max_samples]

    max_abs = waveform.abs().max()
    if max_abs > 0:
        waveform = waveform / max_abs

    return waveform


class AudioMetadataDataset(Dataset):
    """
    Simple audio metadata dataset for feature extraction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        max_seconds: Optional[float] = 30.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        waveform = load_audio(
            row["audio_path"],
            sample_rate=self.sample_rate,
            max_seconds=self.max_seconds,
        )

        return {
            "index": idx,
            "waveform": waveform,
            "label": float(row["label"]),
            "audio_path": row["audio_path"],
            "subject_id": row["subject_id"],
        }


def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-length waveforms.
    """

    max_len = max(item["waveform"].numel() for item in batch)

    waveforms = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["waveform"].numel()
        waveforms[i, :length] = item["waveform"]
        attention_mask[i, :length] = 1

    return {
        "index": [item["index"] for item in batch],
        "waveform": waveforms,
        "attention_mask": attention_mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "audio_path": [item["audio_path"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
    }


class Wav2Vec2FeatureExtractor:
    """
    Hugging Face wav2vec2-style feature extractor.

    This class uses AutoModel from transformers and mean-pools the last hidden
    state. It can also work with similar speech SSL models if their forward
    method returns last_hidden_state.
    """

    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-base",
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        pooling: str = "mean",
    ) -> None:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for audio_embedding features. "
                "Install it with: pip install transformers"
            ) from exc

        self.checkpoint = checkpoint
        self.device = device or torch.device("cpu")
        self.sample_rate = sample_rate
        self.pooling = pooling

        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_batch(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract embeddings for a batch of waveforms.

        Parameters
        ----------
        waveforms:
            [B, T]
        attention_mask:
            [B, T]

        Returns
        -------
        embeddings:
            [B, D]
        """

        waveforms = waveforms.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_values=waveforms,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state  # [B, N, D]

        if self.pooling == "mean":
            embedding = hidden.mean(dim=1)
        elif self.pooling == "max":
            embedding = hidden.max(dim=1).values
        elif self.pooling == "cls":
            embedding = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return embedding.detach().cpu()


def cache_path_for_audio(
    audio_path: str,
    cache_dir: Path,
) -> Path:
    """
    Create a stable cache path for one audio file.
    """

    safe_name = str(audio_path).replace("/", "_").replace("\\", "_").replace(":", "_")
    return cache_dir / f"{safe_name}.pt"


def extract_audio_embeddings(
    df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device,
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract or load cached audio embeddings.

    Returns
    -------
    X:
        Feature matrix [N, D]
    y:
        Labels [N]
    """

    features_cfg = config.get("features", {})
    audio_cfg = features_cfg.get("audio_embedding", {})

    checkpoint = audio_cfg.get("checkpoint", "facebook/wav2vec2-base")
    pooling = audio_cfg.get("pooling", "mean")
    cache_dir = Path(audio_cfg.get("cache_dir", "outputs/features/wav2vec2"))
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    max_seconds = audio_cfg.get("max_audio_seconds", 30.0)
    batch_size = int(audio_cfg.get("batch_size", 8))
    num_workers = int(audio_cfg.get("num_workers", 2))

    ensure_dir(cache_dir)

    dataset = AudioMetadataDataset(
        df=df,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=audio_collate_fn,
    )

    extractor = Wav2Vec2FeatureExtractor(
        checkpoint=checkpoint,
        device=device,
        sample_rate=sample_rate,
        pooling=pooling,
    )

    features: List[Optional[torch.Tensor]] = [None] * len(df)
    labels = df["label"].astype(float).to_numpy()

    for batch_idx, batch in enumerate(loader):
        uncached_indices = []
        uncached_waveforms = []
        uncached_masks = []

        for local_i, idx in enumerate(batch["index"]):
            audio_path = batch["audio_path"][local_i]
            cpath = cache_path_for_audio(audio_path, cache_dir)

            if cpath.exists():
                features[idx] = torch.load(cpath, map_location="cpu")
            else:
                uncached_indices.append(idx)
                uncached_waveforms.append(batch["waveform"][local_i])
                uncached_masks.append(batch["attention_mask"][local_i])

        if len(uncached_indices) > 0:
            waveforms = torch.stack(uncached_waveforms, dim=0)
            masks = torch.stack(uncached_masks, dim=0)

            embeddings = extractor.extract_batch(
                waveforms=waveforms,
                attention_mask=masks,
            )

            for local_i, idx in enumerate(uncached_indices):
                emb = embeddings[local_i]
                features[idx] = emb

                audio_path = df.iloc[idx]["audio_path"]
                cpath = cache_path_for_audio(audio_path, cache_dir)
                torch.save(emb, cpath)

        if (batch_idx + 1) % int(audio_cfg.get("log_interval", 20)) == 0:
            logger.info(
                f"Extracted/cached features for batch "
                f"{batch_idx + 1}/{len(loader)}"
            )

    if any(item is None for item in features):
        missing = sum(item is None for item in features)
        raise RuntimeError(f"{missing} feature vectors were not extracted.")

    X = torch.stack([item for item in features if item is not None], dim=0).numpy()
    y = labels.astype(np.float32)

    return X, y


def extract_handcrafted_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for handcrafted/openSMILE feature extraction.

    For a clean public repo, this function provides a simple fallback using
    basic waveform statistics. Users can replace it with openSMILE eGeMAPS
    extraction later.
    """

    features_cfg = config.get("features", {})
    hand_cfg = features_cfg.get("handcrafted", {})

    sample_rate = int(hand_cfg.get("sample_rate", 16000))
    max_seconds = hand_cfg.get("max_audio_seconds", 30.0)

    X: List[List[float]] = []
    y: List[float] = []

    for i, row in df.iterrows():
        waveform = load_audio(
            row["audio_path"],
            sample_rate=sample_rate,
            max_seconds=max_seconds,
        )

        x = waveform.detach().cpu().numpy()

        # Basic signal statistics as a lightweight placeholder.
        feature = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.mean(np.abs(x))),
            float(np.sqrt(np.mean(x**2) + 1e-8)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 50)),
            float(np.percentile(x, 75)),
        ]

        X.append(feature)
        y.append(float(row["label"]))

        if (i + 1) % int(hand_cfg.get("log_interval", 100)) == 0:
            logger.info(f"Extracted handcrafted features for {i + 1} samples")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def extract_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device,
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features based on config.
    """

    feature_type = config.get("features", {}).get("type", "audio_embedding")

    if feature_type == "audio_embedding":
        return extract_audio_embeddings(
            df=df,
            config=config,
            device=device,
            logger=logger,
        )

    if feature_type == "handcrafted":
        return extract_handcrafted_features(
            df=df,
            config=config,
            logger=logger,
        )

    raise ValueError(f"Unknown feature type: {feature_type}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and Spearman correlation.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        rho = 0.0
    else:
        rho_result = spearmanr(y_true, y_pred)
        rho = float(rho_result.correlation)
        if np.isnan(rho):
            rho = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(rho),
    }


# ---------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------


def run_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Train and evaluate Ridge Regression.
    """

    ridge_cfg = config.get("regressors", {}).get("ridge", {})
    alpha = float(ridge_cfg.get("alpha", 1.0))

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    return model.predict(X_test)


def run_xgboost_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Train and evaluate XGBoost Regressor.
    """

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the XGBoost baseline. "
            "Install it with: pip install xgboost"
        ) from exc

    xgb_cfg = config.get("regressors", {}).get("xgboost", {})

    model = XGBRegressor(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 4)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        objective=xgb_cfg.get("objective", "reg:squarederror"),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
        n_jobs=int(xgb_cfg.get("n_jobs", 4)),
    )

    model.fit(X_train, y_train)

    return model.predict(X_test)


class MLPRegressorTorch(nn.Module):
    """
    Simple two-layer MLP regressor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: List[nn.Module] = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(act_layer())
            layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_mlp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """
    Train and evaluate MLP regressor.
    """

    mlp_cfg = config.get("regressors", {}).get("mlp", {})
    train_cfg = mlp_cfg.get("training", {})

    input_dim = X_train.shape[1]
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128])
    dropout = float(mlp_cfg.get("dropout", 0.3))
    activation = mlp_cfg.get("activation", "relu")

    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    early_stopping = bool(train_cfg.get("early_stopping", True))
    patience = int(train_cfg.get("patience", 10))

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MLPRegressorTorch(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for _epoch in range(epochs):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(losses))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if early_stopping and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred = model(X_test_tensor).detach().cpu().numpy()

    return pred


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------


def run_enabled_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Train enabled regressors and compute metrics.

    Returns
    -------
    metrics:
        Dict of metrics by regressor name.
    prediction_rows:
        Rows for prediction CSV.
    """

    regressors_cfg = config.get("regressors", {})

    metrics: Dict[str, Dict[str, float]] = {}
    prediction_rows: List[Dict[str, Any]] = []

    if regressors_cfg.get("ridge", {}).get("enabled", True):
        pred = run_ridge_regression(X_train, y_train, X_test, config)
        metrics["ridge"] = compute_regression_metrics(y_test, pred)

        prediction_rows.extend(
            build_prediction_rows("ridge", y_test, pred)
        )

    if regressors_cfg.get("xgboost", {}).get("enabled", True):
        pred = run_xgboost_regression(X_train, y_train, X_test, config)
        metrics["xgboost"] = compute_regression_metrics(y_test, pred)

        prediction_rows.extend(
            build_prediction_rows("xgboost", y_test, pred)
        )

    if regressors_cfg.get("mlp", {}).get("enabled", True):
        pred = run_mlp_regression(X_train, y_train, X_test, config, device)
        metrics["mlp"] = compute_regression_metrics(y_test, pred)

        prediction_rows.extend(
            build_prediction_rows("mlp", y_test, pred)
        )

    return metrics, prediction_rows


def build_prediction_rows(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Build prediction CSV rows.
    """

    rows = []

    for i, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        rows.append(
            {
                "model": model_name,
                "index": i,
                "y_true": float(true_value),
                "y_pred": float(pred_value),
                "absolute_error": float(abs(true_value - pred_value)),
            }
        )

    return rows


def standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using training-set statistics only.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def run_downstream_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full downstream regression evaluation.

    Parameters
    ----------
    config:
        Loaded downstream.yaml config.

    Returns
    -------
    dict
        Summary with metrics and output paths.
    """

    experiment_cfg = config.get("experiment", {})
    output_cfg = config.get("output", {})

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(
        experiment_cfg.get(
            "output_dir",
            "outputs/metrics/downstream",
        )
    )
    ensure_dir(output_dir)

    logger = setup_logger(
        name="evaluation.downstream",
        log_file=output_dir / "downstream.log",
    )

    logger.info("Starting downstream regression evaluation")
    logger.info(f"Output directory: {output_dir}")

    save_yaml(config, output_dir / "downstream_config.yaml")

    device = get_device_from_config(config)
    logger.info(f"Using device: {device}")

    train_df, test_df = build_train_test_metadata(config)

    logger.info(f"Training rows: {len(train_df)}")
    logger.info(f"Test rows: {len(test_df)}")
    logger.info(
        f"Train with synthetic: "
        f"{config.get('dataset', {}).get('train_with_synthetic', True)}"
    )

    logger.info("Extracting training features")
    X_train, y_train = extract_features(
        df=train_df,
        config=config,
        device=device,
        logger=logger,
    )

    logger.info("Extracting test features")
    X_test, y_test = extract_features(
        df=test_df,
        config=config,
        device=device,
        logger=logger,
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    X_train, X_test, scaler = standardize_features(X_train, X_test)

    metrics, prediction_rows = run_enabled_regressors(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
        device=device,
    )

    prediction_file = output_dir / output_cfg.get(
        "prediction_file",
        "predictions.csv",
    )
    summary_file = output_dir / output_cfg.get(
        "summary_file",
        "summary.json",
    )

    if config.get("evaluation", {}).get("save_predictions", True):
        save_csv_rows(
            rows=prediction_rows,
            path=prediction_file,
            fieldnames=[
                "model",
                "index",
                "y_true",
                "y_pred",
                "absolute_error",
            ],
        )

    summary = {
        "metrics": metrics,
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
        "feature_type": config.get("features", {}).get("type", "audio_embedding"),
        "train_with_synthetic": bool(
            config.get("dataset", {}).get("train_with_synthetic", True)
        ),
        "prediction_file": str(prediction_file),
        "summary_file": str(summary_file),
    }

    if config.get("evaluation", {}).get("save_summary", True):
        save_json(summary, summary_file)

    logger.info("Downstream evaluation complete")

    for model_name, model_metrics in metrics.items():
        logger.info(
            f"{model_name}: "
            f"MAE={model_metrics['mae']:.4f}, "
            f"RMSE={model_metrics['rmse']:.4f}, "
            f"Spearman={model_metrics['spearman']:.4f}"
        )

    return summary