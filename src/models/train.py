"""

The main entry point is:

    train_lflow(config)

This file is intended to be imported by:
    src/scripts/train_lflow.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from .data import LongitudinalSpeechDataset
from .generator import build_lflow_model_from_config
from .losses import build_loss_from_config
from .preprocessing import (
    LFlowTrainingTransform,
    build_preprocessor_from_config,
    lflow_training_collate_fn,
)
from .utils import (
    MetricTracker,
    build_optimizer_from_config,
    build_scheduler_from_config,
    count_parameters,
    ensure_dir,
    find_latest_checkpoint,
    format_seconds,
    get_current_lr,
    get_device_from_config,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    save_run_summary,
    save_yaml,
    set_seed,
    setup_logger,
)


# ---------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------


def build_train_val_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], Any]:
    """
    Build training and validation dataloaders.

    The dataset returns one target session and its longitudinal context.
    The transform loads audio, extracts mel-spectrograms, creates masks,
    text tokens, flow-matching noisy paths, and context tensors.

    Returns
    -------
    train_loader:
        Training dataloader.
    val_loader:
        Validation dataloader or None.
    preprocessor:
        LFlowPreprocessor instance. Its tokenizer is needed to obtain vocab size.
    """

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    validation_cfg = config.get("validation", {})

    preprocessor = build_preprocessor_from_config(config)

    train_transform = LFlowTrainingTransform(
        preprocessor=preprocessor,
        random_crop=True,
        load_context_audio=True,
    )

    dataset = LongitudinalSpeechDataset(
        metadata_path=dataset_cfg["metadata_path"],
        split_path=dataset_cfg.get("split_path"),
        split=dataset_cfg.get("train_split", "train"),
        root_dir=dataset_cfg.get("root_dir"),
        subject_column=dataset_cfg.get("subject_column", "subject_id"),
        audio_column=dataset_cfg.get("audio_column", "audio_path"),
        transcript_column=dataset_cfg.get("transcript_column", "transcript"),
        time_column=dataset_cfg.get("time_column", "timestamp"),
        label_column=dataset_cfg.get("label_column", "label"),
        min_sessions=dataset_cfg.get("min_sessions", 2),
        target_sampling=dataset_cfg.get("target_sampling", "all"),
        transform=train_transform,
    )

    val_loader = None

    if validation_cfg.get("enabled", True):
        val_ratio = float(validation_cfg.get("val_ratio", 0.1))

        if 0.0 < val_ratio < 1.0 and len(dataset) > 1:
            val_size = max(1, int(len(dataset) * val_ratio))
            train_size = len(dataset) - val_size

            generator = torch.Generator().manual_seed(
                int(config.get("experiment", {}).get("seed", 42))
            )

            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=generator,
            )
        else:
            train_dataset = dataset
            val_dataset = None
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=bool(training_cfg.get("drop_last", True)),
        collate_fn=lflow_training_collate_fn,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(training_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(training_cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=False,
            collate_fn=lflow_training_collate_fn,
        )

    return train_loader, val_loader, preprocessor


# ---------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: Any,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    """

    model.train()

    training_cfg = config.get("training", {})
    log_interval = int(config.get("experiment", {}).get("log_interval", 50))

    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    mixed_precision = bool(training_cfg.get("mixed_precision", True))

    tracker = MetricTracker()

    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()

    for step, batch in enumerate(train_loader, start=1):
        batch = move_to_device(batch, device)

        use_amp = mixed_precision and device.type == "cuda"

        with autocast(enabled=use_amp):
            outputs = model(batch)

            losses = criterion(
                predicted_vector_field=outputs["predicted_vector_field"],
                target_vector_field=batch["target_vector_field"],
                z_prog=outputs["z_prog"],
                target_time=batch["target_time"],
                target_label=batch["target_label"],
                mask=batch.get("mask"),
            )

            loss = losses["loss"] / grad_accum_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = step % grad_accum_steps == 0 or step == len(train_loader)

        if should_step:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        batch_size = len(batch["subject_id"])

        tracker.update(
            {
                "loss": losses["loss"],
                "loss_gen": losses["loss_gen"],
                "loss_prog": losses["loss_prog"],
                "loss_ord": losses["loss_ord"],
            },
            n=batch_size,
        )

        if step % log_interval == 0:
            elapsed = format_seconds(time.time() - start_time)
            lr = get_current_lr(optimizer)

            logger.info(
                f"Epoch [{epoch}] "
                f"Step [{step}/{len(train_loader)}] "
                f"LR {lr:.6e} | "
                f"{tracker.format()} | "
                f"Elapsed {elapsed}"
            )

    return tracker.averages()


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: Any,
) -> Dict[str, float]:
    """
    Validate for one epoch.
    """

    model.eval()

    tracker = MetricTracker()

    for batch in val_loader:
        batch = move_to_device(batch, device)

        outputs = model(batch)

        losses = criterion(
            predicted_vector_field=outputs["predicted_vector_field"],
            target_vector_field=batch["target_vector_field"],
            z_prog=outputs["z_prog"],
            target_time=batch["target_time"],
            target_label=batch["target_label"],
            mask=batch.get("mask"),
        )

        batch_size = len(batch["subject_id"])

        tracker.update(
            {
                "loss": losses["loss"],
                "loss_gen": losses["loss_gen"],
                "loss_prog": losses["loss_prog"],
                "loss_ord": losses["loss_ord"],
            },
            n=batch_size,
        )

    metrics = tracker.averages()
    logger.info(f"Epoch [{epoch}] Validation | {tracker.format(prefix='val_')}")

    return metrics


# ---------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------


def train_lflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train L-Flow from a loaded config dictionary.

    Parameters
    ----------
    config:
        Loaded YAML config.

    Returns
    -------
    dict
        Training summary.
    """

    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(experiment_cfg.get("output_dir", "outputs/runs/lflow"))
    checkpoint_dir = Path(
        experiment_cfg.get("checkpoint_dir", "checkpoints/lflow")
    )

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)

    logger = setup_logger(
        name="lflow.train",
        log_file=output_dir / "train.log",
    )

    logger.info("Starting L-Flow training")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    save_yaml(config, output_dir / "config.yaml")

    device = get_device_from_config(config)
    logger.info(f"Using device: {device}")

    train_loader, val_loader, preprocessor = build_train_val_dataloaders(config)

    vocab_size = preprocessor.tokenizer.vocab_size

    model = build_lflow_model_from_config(
        config=config,
        vocab_size=vocab_size,
    )
    model = model.to(device)

    logger.info(
        f"Model parameters: "
        f"{count_parameters(model, trainable_only=True):,} trainable / "
        f"{count_parameters(model, trainable_only=False):,} total"
    )

    criterion = build_loss_from_config(config)
    criterion = criterion.to(device)

    optimizer = build_optimizer_from_config(model, config)

    epochs = int(training_cfg.get("epochs", 100))
    total_steps = epochs * len(train_loader)

    scheduler = build_scheduler_from_config(
        optimizer=optimizer,
        config=config,
        total_steps=total_steps,
    )

    mixed_precision = bool(training_cfg.get("mixed_precision", True))
    scaler = GradScaler(enabled=mixed_precision and device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    resume_from = experiment_cfg.get("resume_from")

    if resume_from == "latest":
        latest = find_latest_checkpoint(checkpoint_dir)
        resume_from = str(latest) if latest is not None else None

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")

        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            strict=True,
        )

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_metric", best_val_loss))

        logger.info(
            f"Resumed at epoch {start_epoch}, "
            f"global step {global_step}, "
            f"best val loss {best_val_loss:.6f}"
        )

    save_run_summary(
        path=output_dir / "run_summary.json",
        config=config,
        model=model,
        extra={
            "vocab_size": vocab_size,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader) if val_loader is not None else 0,
        },
    )

    save_interval = int(experiment_cfg.get("save_interval", 5))
    val_interval = int(config.get("validation", {}).get("val_interval", 1))

    training_start = time.time()

    history = {
        "train": [],
        "val": [],
    }

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            config=config,
            logger=logger,
            scaler=scaler,
        )

        global_step += len(train_loader)

        train_record = {
            "epoch": epoch,
            **train_metrics,
            "lr": get_current_lr(optimizer),
        }
        history["train"].append(train_record)

        val_metrics = None

        if val_loader is not None and epoch % val_interval == 0:
            val_metrics = validate_one_epoch(
                model=model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                logger=logger,
            )

            val_record = {
                "epoch": epoch,
                **val_metrics,
            }
            history["val"].append(val_record)

            val_loss = float(val_metrics["loss"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_val_loss,
                    config=config,
                    extra={
                        "vocab_size": vocab_size,
                        "val_metrics": val_metrics,
                    },
                )

                logger.info(
                    f"Saved new best checkpoint to {best_path} "
                    f"with val loss {best_val_loss:.6f}"
                )

        should_save = epoch % save_interval == 0 or epoch == epochs

        if should_save:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_val_loss,
                config=config,
                extra={
                    "vocab_size": vocab_size,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

            logger.info(f"Saved checkpoint to {ckpt_path}")

        epoch_time = format_seconds(time.time() - epoch_start)

        logger.info(
            f"Finished epoch {epoch}/{epochs} | "
            f"Train loss {train_metrics['loss']:.6f} | "
            f"Best val loss {best_val_loss:.6f} | "
            f"Epoch time {epoch_time}"
        )

    total_time = format_seconds(time.time() - training_start)

    summary = {
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "global_step": global_step,
        "total_time": total_time,
        "history": history,
    }

    logger.info(f"Training finished in {total_time}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return summary


# ---------------------------------------------------------------------
# Optional convenience function for direct use
# ---------------------------------------------------------------------


def train_from_config_path(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config and train L-Flow.
    """

    from .utils import load_yaml

    config = load_yaml(config_path)
    return train_lflow(config)


# ---------------------------------------------------------------------
# Quick smoke test with synthetic tensors is not included here.
# Real training requires metadata and audio files.
# ---------------------------------------------------------------------