"""Trainer: Configurable training loop with callback integration.This module implements a flexible and lightweight `Trainer` class designed
for research workflows. It handles the full training and validation process,
integrates seamlessly with callbacks, EMA (exponential moving average),
learning rate schedulers, and gradient clipping, and logs progress using
Hydra’s centralized logging system.

The design emphasizes:
    • Simplicity — minimal boilerplate while remaining extensible.
    • Transparency — callbacks receive lifecycle events at key stages.
    • Flexibility — supports gradient accumulation, checkpoint resume,
      and multiple model architectures.

Typical usage example
---------------------
>>> trainer = Trainer(
...     model=model,
...     optimizer=optimizer,
...     train_config=cfg.trainer.train_config,
...     data_wrapper=wrapper,
...     output_dir=cfg.trainer.output_dir,
...     scheduler=scheduler,
...     ema=ema,
...     callbacks=callbacks,
... )
>>> trainer.train(train_loader, validation_loader)

The model must implement at least one of the following:
    • `train_step(batch_dict)` -> returns a scalar loss or a dict containing "loss"
    • (optional) `loss(pred, target)` if additional flexibility is desired

Callbacks can extend training behavior by overriding any of the following hooks:
    - on_train_begin(self, trainer)
    - on_epoch_begin(self, trainer)
    - on_batch_end(self, trainer)
    - on_validation_end(self, trainer)
    - on_epoch_end(self, trainer)
    - on_train_end(self, trainer)
    - (optional) resume(self, trainer)

All logging goes through the standard `logging` module, which Hydra automatically
redirects into timestamped log directories (e.g. `outputs/YYYY-MM-DD/HH-MM-SS/`).
"""

import os
import torch
import logging
from torch.utils.data import DataLoader
from typing import Optional, List
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

log = logging.getLogger(__name__)




class Trainer:
    """Encapsulates a complete training and validation loop with callback hooks.

    The `Trainer` class coordinates the iterative training process over epochs,
    handling forward/backward passes, optimizer steps, learning rate scheduling,
    EMA updates, and validation evaluation. It also provides lifecycle signals
    to callback objects, enabling modular logging, checkpointing, and monitoring.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained. Must implement a `train_step(batch_dict)` method
        that returns a loss tensor or a dictionary containing the key `"loss"`.

    optimizer : torch.optim.Optimizer
        Optimizer instance responsible for model parameter updates.

    train_config : omegaconf.DictConfig or similar
        Training configuration with at least the following attributes:
            - `n_epoch`: number of epochs to train
            - `device`: computation device (e.g. "cuda" or "cpu")
            - `clip_grad_norm`: optional gradient clipping threshold
            - `eval_patience`: evaluation frequency in epochs
            - `start_epoch`: epoch to start from (for resume)
            - (optional) `accumulate_grad_batches`: steps to accumulate gradients

    data_wrapper : callable
        Function that transforms a raw batch from the DataLoader into the
        ordered dictionary expected by the model (e.g. `{"x": ..., "y": ...}`).

    output_dir : str
        Base directory where logs, checkpoints, and results are stored.

    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler, stepped at the end of each epoch.

    ema : torch.nn.Module, optional
        Exponential moving average tracker with `register()` and `update()` methods.

    callbacks : list of Callback, optional
        List of callback instances. Each will be notified at defined training events.

    Notes
    -----
    - All logging is handled via Hydra's logger (`logging.getLogger(__name__)`).
    - Checkpointing and resuming are typically managed via `CheckpointCallback`.
    - Gradient clipping, accumulation, and EMA updates are applied automatically.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_config,
        data_wrapper,
        output_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ema: Optional[torch.nn.Module] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize trainer state and runtime configuration."""
        self._init_config(train_config)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.output_dir = output_dir

        self.data_wrapper = data_wrapper
        self.callbacks = callbacks if callbacks is not None else []

        self._stop_training = False

        self.scaler = GradScaler(enabled=self.use_amp)
        

    def _init_config(self, train_config):
        """Extract relevant attributes from the Hydra configuration object."""
        self.n_epoch = train_config.n_epoch
        self.device = train_config.device
        self.clip_grad_norm = train_config.clip_grad_norm
        self.eval_patience = train_config.eval_patience
        self.current_epoch = train_config.start_epoch
        self.accumulate_grad_batches = getattr(
            train_config, "accumulate_grad_batches", 1
        )
        self.use_amp = train_config.use_amp
        self.use_tqdm = train_config.use_tqdm

    def _progress(self, iterable, **kwargs):
        """Conditional tqdm wrapper."""
        if self.use_tqdm:
            return tqdm(iterable, **kwargs)
        else:
            return iterable

    def train(self, train_loader: DataLoader, validation_loader: Optional[DataLoader] = None):
        """Execute the main training loop.

        This function orchestrates the full training and validation process,
        including optimizer updates, EMA, learning rate scheduling, and callback
        notifications.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader yielding training batches.

        validation_loader : torch.utils.data.DataLoader, optional
            DataLoader yielding validation batches for periodic evaluation.
        """
        if self.ema is not None:
            self.ema.register(self.model)
        self.model.train()

        # Allow callbacks to restore checkpointed state if applicable
        for cb in self.callbacks:
            if hasattr(cb, "resume"):
                cb.resume(self)

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.current_epoch, self.n_epoch):
            self.current_epoch = epoch
            self._stop_training = False
            for cb in self.callbacks:
                cb.on_epoch_begin(self)

            epoch_loss = 0.0
            self.model.train()

            for batch_idx, batch in enumerate(
                self._progress(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epoch}")
            ):

                self.current_batch = self.data_wrapper(batch)

                # Forward + loss computation
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=self.use_amp):
                    loss_out = self.model.train_step(self.current_batch)
                    loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out

                    if not torch.isfinite(loss):
                        print(f"[NaN] Non-finite loss at epoch {epoch}, {loss.item()}")
                        # Option 1: skip this batch
                        continue

                self.loss_batch = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "value": float(loss.item()),
                }

                # Backward
                scaled_loss = loss / self.accumulate_grad_batches
                self.scaler.scale(scaled_loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.clip_grad_norm and self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.ema is not None:
                        self.ema.update(self.model)

                epoch_loss += loss.item()
                for cb in self.callbacks:
                    cb.on_batch_end(self)

            self.last_train_loss = epoch_loss / len(train_loader)
            log.info(f"[Trainer] Epoch {epoch}: train_loss={self.last_train_loss:.6f}")

            # Validation
            if epoch % self.eval_patience == 0 and validation_loader is not None:
                self.val_loss = self.eval(validation_loader)
            else:
                self.val_loss = None

            for cb in self.callbacks:
                cb.on_validation_end(self)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.val_loss)
                else:
                    self.scheduler.step()

            for cb in self.callbacks:
                cb.on_epoch_end(self)

            if self._stop_training:
                log.info("[Trainer] Early stopping triggered.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def eval(self, validation_loader: DataLoader) -> float:
        """Evaluate model performance on a validation set.

        Parameters
        ----------
        validation_loader : torch.utils.data.DataLoader
            DataLoader yielding validation batches.

        Returns
        -------
        float
            Mean validation loss across all batches.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad(), autocast(enabled=self.use_amp):
            for batch in self._progress(
                validation_loader,
                desc=f"Validation: Epoch {self.current_epoch + 1}/{self.n_epoch}",
            ):
                dict_ord = self.data_wrapper(batch)
                if hasattr(self.model, "validation_step"):
                    loss_out = self.model.validation_step(dict_ord)
                else:
                    loss_out = self.model.train_step(dict_ord)

                loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out
                total_loss += loss.item()

        mean_loss = total_loss / len(validation_loader)
        log.info(f"[Trainer] Validation loss: {mean_loss:.6f}")

        self.model.train()  # restore train mode after validation
        return mean_loss
