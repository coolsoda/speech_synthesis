"""
This module includes:
1. YAML/JSON config loading.
2. Random seed control.
3. Device setup.
4. Directory creation.
5. Checkpoint saving/loading.
6. Metric logging.
7. Tensor/device helpers.
8. JSON/CSV writing helpers.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import yaml


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to YAML file.

    Returns
    -------
    dict
        Loaded configuration.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def save_yaml(config: Mapping[str, Any], path: PathLike) -> None:
    """
    Save a dictionary as a YAML file.
    """

    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(config),
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def load_json(path: PathLike) -> Dict[str, Any]:
    """
    Load a JSON file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    path: PathLike,
    indent: int = 2,
) -> None:
    """
    Save data to a JSON file.
    """

    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_nested(
    config: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Safely retrieve a nested config value.

    Example
    -------
    get_nested(config, ["model", "generator", "hidden_dim"], 512)
    """

    current: Any = config

    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]

    return current


def update_nested(
    config: Dict[str, Any],
    keys: Sequence[str],
    value: Any,
) -> Dict[str, Any]:
    """
    Update a nested dictionary value in-place and return the dictionary.
    """

    if len(keys) == 0:
        raise ValueError("keys must not be empty.")

    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
    return config


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(
    seed: int,
    deterministic: bool = False,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Random seed.
    deterministic:
        If True, request deterministic PyTorch behavior where possible.
        This may reduce performance.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------


def get_device(
    use_cuda: bool = True,
    cuda_device: int = 0,
) -> torch.device:
    """
    Get a torch.device from config options.
    """

    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")

    return torch.device("cpu")


def get_device_from_config(config: Mapping[str, Any]) -> torch.device:
    """
    Build torch.device from config.
    """

    device_cfg = config.get("device", {})

    use_cuda = bool(device_cfg.get("use_cuda", True))
    cuda_device = int(device_cfg.get("cuda_device", 0))

    return get_device(
        use_cuda=use_cuda,
        cuda_device=cuda_device,
    )


def move_to_device(
    obj: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """
    Recursively move tensors in an object to device.

    Supports:
    - torch.Tensor
    - dict
    - list
    - tuple

    Non-tensor values are returned unchanged.
    """

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {
            key: move_to_device(value, device, non_blocking=non_blocking)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(value, device, non_blocking=non_blocking)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(value, device, non_blocking=non_blocking)
            for value in obj
        )

    return obj


def detach_to_cpu(obj: Any) -> Any:
    """
    Recursively detach tensors and move them to CPU.

    Useful before logging, saving predictions, or converting to NumPy.
    """

    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {key: detach_to_cpu(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [detach_to_cpu(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(detach_to_cpu(value) for value in obj)

    return obj


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """
    Count model parameters.
    """

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------
# Filesystem utilities
# ---------------------------------------------------------------------


def ensure_dir(path: PathLike) -> Path:
    """
    Create directory if it does not exist.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: PathLike) -> Path:
    """
    Remove and recreate a directory.

    Use carefully.
    """

    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: PathLike, dst: PathLike) -> None:
    """
    Copy file from src to dst, creating parent directories if needed.
    """

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def timestamp_string() -> str:
    """
    Return a compact timestamp string for run naming.
    """

    return time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------


def setup_logger(
    name: str = "lflow",
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a console/file logger.

    Parameters
    ----------
    name:
        Logger name.
    log_file:
        Optional file path to save logs.
    level:
        Logging level.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers in interactive environments.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Track running average of scalar values.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.total = 0.0
        self.count = 0
        self.avg = 0.0

    def update(
        self,
        value: Union[float, torch.Tensor],
        n: int = 1,
    ) -> None:
        if torch.is_tensor(value):
            value = float(value.detach().cpu())

        self.value = float(value)
        self.total += float(value) * n
        self.count += n
        self.avg = self.total / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.6f}"


class MetricTracker:
    """
    Track multiple scalar metrics.
    """

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(
        self,
        metrics: Mapping[str, Union[float, torch.Tensor]],
        n: int = 1,
    ) -> None:
        for key, value in metrics.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key)
            self.meters[key].update(value, n=n)

    def averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def format(self, prefix: str = "") -> str:
        parts = []
        for key, meter in self.meters.items():
            metric_name = f"{prefix}{key}" if prefix else key
            parts.append(f"{metric_name}: {meter.avg:.6f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------


def save_checkpoint(
    path: PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.
    """

    path = Path(path)
    ensure_dir(path.parent)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        try:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        except AttributeError:
            pass

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if step is not None:
        checkpoint["step"] = step

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    if config is not None:
        checkpoint["config"] = dict(config)

    if extra is not None:
        checkpoint["extra"] = dict(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: PathLike,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and optionally restore model/optimizer/scheduler states.

    Returns
    -------
    dict
        Full checkpoint dictionary.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    if model is not None:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: PathLike) -> Optional[Path]:
    """
    Find the latest checkpoint file in a directory.

    This looks for files ending in .pt or .pth and returns the most recently
    modified one.
    """

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    candidates = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


# ---------------------------------------------------------------------
# Optimizer and scheduler helpers
# ---------------------------------------------------------------------


def build_optimizer_from_config(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer from training config.
    """

    training_cfg = config.get("training", {})
    opt_cfg = training_cfg.get("optimizer", {})

    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    betas = opt_cfg.get("betas", [0.9, 0.999])

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_steps: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Build learning-rate scheduler from config.

    Supported:
    - none
    - cosine
    - step
    """

    training_cfg = config.get("training", {})
    sched_cfg = training_cfg.get("scheduler", {})

    name = str(sched_cfg.get("name", "none")).lower()

    if name in {"none", "null"}:
        return None

    if name == "cosine":
        if total_steps is None:
            total_steps = int(sched_cfg.get("total_steps", 10000))

        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))

        return WarmupCosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )

    if name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unknown scheduler: {name}")


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Linear warmup followed by cosine decay.

    This scheduler keeps track of the initial learning rates from the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")

        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        step = max(self.last_epoch, 0)

        lrs = []

        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                lr = base_lr * float(step + 1) / float(self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(
                    1,
                    self.total_steps - self.warmup_steps,
                )
                progress = min(max(progress, 0.0), 1.0)

                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine

            lrs.append(float(lr))

        return lrs


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Return the learning rate of the first optimizer parameter group.
    """

    return float(optimizer.param_groups[0]["lr"])


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------


def save_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    path: PathLike,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """
    Save a list of dictionaries to CSV.

    Parameters
    ----------
    rows:
        Sequence of row dictionaries.
    path:
        Output CSV path.
    fieldnames:
        Optional explicit column order.
    """

    path = Path(path)
    ensure_dir(path.parent)

    if fieldnames is None:
        fieldnames = infer_fieldnames(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(dict(row))


def append_csv_row(
    row: Mapping[str, Any],
    path: PathLike,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """
    Append one row to a CSV file.

    If the file does not exist, write the header first.
    """

    path = Path(path)
    ensure_dir(path.parent)

    file_exists = path.exists()

    if fieldnames is None:
        fieldnames = list(row.keys())

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(dict(row))


def infer_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """
    Infer CSV fieldnames from a list of row dictionaries.
    """

    if len(rows) == 0:
        return []

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    return fieldnames


# ---------------------------------------------------------------------
# Tensor shape helpers
# ---------------------------------------------------------------------


def assert_shape(
    tensor: torch.Tensor,
    expected_ndim: Optional[int] = None,
    name: str = "tensor",
) -> None:
    """
    Lightweight tensor dimension check.
    """

    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} expected ndim={expected_ndim}, got shape={tuple(tensor.shape)}"
        )


def remove_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Remove 'module.' prefix from DataParallel checkpoints.
    """

    output = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        output[key] = value

    return output


def add_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Add 'module.' prefix to checkpoint keys.
    """

    output = {}

    for key, value in state_dict.items():
        if not key.startswith("module."):
            key = f"module.{key}"
        output[key] = value

    return output


# ---------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------


def format_seconds(seconds: float) -> str:
    """
    Convert seconds into a human-readable string.
    """

    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"

    if minutes > 0:
        return f"{minutes:d}m {seconds:02d}s"

    return f"{seconds:d}s"


def format_number(num: Union[int, float]) -> str:
    """
    Format large numbers with commas.
    """

    if isinstance(num, int):
        return f"{num:,}"

    return f"{num:,.4f}"


# ---------------------------------------------------------------------
# Small run summary helper
# ---------------------------------------------------------------------


def save_run_summary(
    path: PathLike,
    config: Mapping[str, Any],
    model: Optional[torch.nn.Module] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save a compact summary of a run.
    """

    summary: Dict[str, Any] = {
        "timestamp": timestamp_string(),
        "config": dict(config),
    }

    if model is not None:
        summary["num_trainable_parameters"] = count_parameters(
            model,
            trainable_only=True,
        )
        summary["num_total_parameters"] = count_parameters(
            model,
            trainable_only=False,
        )

    if extra is not None:
        summary["extra"] = dict(extra)

    save_json(summary, path)