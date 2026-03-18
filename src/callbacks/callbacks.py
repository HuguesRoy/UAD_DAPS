"""Training callbacks utilities.

This module contains lightweight callback classes used by the training
loop. Callbacks provide hooks that are invoked at key points during
training (train begin/end, epoch begin/end, batch end, validation end,
... ). Implementations included are simple examples such as checkpointing,
CSV logging and timing utilities.

Users can extend `Callback` and override the hook methods they need.
"""

import os
import csv
import torch
import logging
from torchvision.utils import save_image
import psutil

log = logging.getLogger(__name__)

class Callback:
    """Base callback class exposing training hooks.

    Subclasses can override any of these methods to receive training
    lifecycle events from the `Trainer`.
    """

    def on_train_begin(self, trainer):
        """Called once before training starts.

        Parameters
        ----------
        trainer : Trainer
            The Trainer instance invoking the callback.
        """
        pass

    def on_epoch_begin(self, trainer):
        """Called at the beginning of each epoch."""
        pass

    def on_batch_end(self, trainer):
        """Called after each training batch is processed."""
        pass

    def on_epoch_end(self, trainer):
        """Called at the end of each epoch."""
        pass

    def on_validation_end(self, trainer):
        """Called after validation has completed for the epoch."""
        pass

    def on_train_end(self, trainer):
        """Called once after training finishes (natural stop or early stop)."""
        pass


class CheckpointCallback(Callback):
    """Callback that saves checkpoints for both Trainer and AdversarialTrainer."""

    def __init__(self, output_dir, every_n_epochs=5, save_best_only=True):
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.save_best_only = save_best_only
        self.best_val_loss = float("inf")

    def _save(self, trainer, folder="checkpoints", tag="checkpoint.pt"):
        """Serialize trainer state to disk."""
        ckpt = {"epoch": trainer.current_epoch}

        # Check trainer type
        if trainer.__class__.__name__ == "AdversarialTrainer":
            ckpt.update(
                {
                    "model": trainer.model.state_dict(),
                    "opt_g": trainer.opt_gen.state_dict(),
                    "opt_d": trainer.opt_disc.state_dict(),
                }
            )
            if getattr(trainer, "scheduler_g", None):
                ckpt["scheduler_g"] = trainer.scheduler_g.state_dict()
            if getattr(trainer, "scheduler_d", None):
                ckpt["scheduler_d"] = trainer.scheduler_d.state_dict()
        else:
            ckpt.update(
                {
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                }
            )
            if getattr(trainer, "scheduler", None):
                ckpt["scheduler"] = trainer.scheduler.state_dict()

        if getattr(trainer, "ema", None):
            ckpt["ema"] = trainer.ema.state_dict()

        # Save to disk
        ckpt_folder = os.path.join(self.output_dir, folder)
        os.makedirs(ckpt_folder, exist_ok=True)
        ckpt_path = os.path.join(ckpt_folder, tag)
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    def on_validation_end(self, trainer):
        if trainer.val_loss is not None and trainer.val_loss < self.best_val_loss:
            self.best_val_loss = trainer.val_loss
            path = self._save(trainer, "best_loss", "best.pt")
            log.info(f"[Checkpoint] New best model saved at {path}")

    def on_epoch_end(self, trainer):
        if trainer.current_epoch % self.every_n_epochs == 0:
            path = self._save(
                trainer, "checkpoints", f"epoch_{trainer.current_epoch}.pt"
            )
            log.info(f"[Checkpoint] Saved checkpoint at {path}")
        
        if trainer.current_epoch == trainer.n_epoch - 1:
            path = self._save(
                trainer, "checkpoints", f"epoch_{trainer.current_epoch}.pt"
            )
            log.info(f"[Checkpoint] Saved final model at {path}")

    def resume(self, trainer):
        """Restore from latest checkpoint."""
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            log.warning(f"[Checkpoint] No checkpoint dir: {ckpt_dir}")
            return

        ckpts = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        ]
        if not ckpts:
            log.warning("[Checkpoint] No checkpoint files found.")
            return

        ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
        ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only = False)

        if trainer.__class__.__name__ == "AdversarialTrainer":
            trainer.model.load_state_dict(ckpt["model"])
            trainer.opt_gen.load_state_dict(ckpt["opt_g"])
            trainer.opt_disc.load_state_dict(ckpt["opt_d"])

            for opt in [trainer.opt_gen, trainer.opt_disc]:
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(trainer.device)

            if getattr(trainer, "scheduler_g", None) and "scheduler_g" in ckpt:
                trainer.scheduler_g.load_state_dict(ckpt["scheduler_g"])
            if getattr(trainer, "scheduler_d", None) and "scheduler_d" in ckpt:
                trainer.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        else:
            trainer.model.load_state_dict(ckpt["model"] )
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
            for state in trainer.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(trainer.device)

            if getattr(trainer, "scheduler", None) and "scheduler" in ckpt:
                trainer.scheduler.load_state_dict(ckpt["scheduler"])

        if getattr(trainer, "ema", None) and "ema" in ckpt:
            trainer.ema.load_state_dict(ckpt["ema"])

        trainer.current_epoch = ckpt.get("epoch", 0) + 1
        log.info(
            f"[Checkpoint] Resumed from {ckpt_path} (epoch {trainer.current_epoch})"
        )

class LoggerCallback(Callback):
    """Logs epoch-level metrics for both Trainer and AdversarialTrainer."""

    def __init__(self, output_dir: str, filename: str = "train_log.csv"):
        self.output_dir = output_dir
        self.filename = filename
        self.filepath = os.path.join(output_dir, filename)
        self._file = None
        self._writer = None

    def on_train_begin(self, trainer):
        import csv

        os.makedirs(self.output_dir, exist_ok=True)

        is_new_run = (trainer.current_epoch == 0) or not os.path.exists(self.filepath)
        mode = "w" if is_new_run else "a"

        self._file = open(self.filepath, mode, newline="")

        if trainer.__class__.__name__ == "AdversarialTrainer":
            fieldnames = ["epoch", "train_loss_g", "train_loss_d", "val_loss"]
        else:
            fieldnames = ["epoch", "train_loss", "val_loss"]

        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)

        if is_new_run:
            self._writer.writeheader()

    def on_epoch_end(self, trainer):
        if trainer.__class__.__name__ == "AdversarialTrainer":
            row = {
                "epoch": trainer.current_epoch,
                "train_loss_g": getattr(trainer, "last_train_loss_g", None),
                "train_loss_d": getattr(trainer, "last_train_loss_d", None),
                "val_loss": getattr(trainer, "val_loss", None),
            }
        else:
            row = {
                "epoch": trainer.current_epoch,
                "train_loss": getattr(trainer, "last_train_loss", None),
                "val_loss": getattr(trainer, "val_loss", None),
            }

        self._writer.writerow(row)
        self._file.flush()
        log.info(f"[Logger] Epoch {trainer.current_epoch}: {row}")

    def on_train_end(self, trainer):
        if self._file is not None:
            self._file.close()

class BatchLossLogger(Callback):
    """
    Logs batch-wise loss values for both Trainer and AdversarialTrainer,
    optionally writing them to CSV.
    """

    def __init__(self, output_dir: str | None = None, log_every: int = 10):
        self.output_dir = output_dir
        self.log_every = log_every
        self.batch_log = []
        self.csv_path = None

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.csv_path = os.path.join(self.output_dir, "batch_loss.csv")

    def on_batch_end(self, trainer):
        info = trainer.loss_batch
        epoch, batch_idx = info["epoch"], info["batch_idx"]

        # Detect trainer type and extract losses
        if trainer.__class__.__name__ == "AdversarialTrainer":
            loss_g = info.get("loss_g")
            loss_d = info.get("loss_d")
            loss_str = f"G: {loss_g:.6f}" if loss_g is not None else ""
            if loss_d is not None:
                loss_str += f" | D: {loss_d:.6f}"
            self.batch_log.append((epoch, batch_idx, loss_g, loss_d))
        else:
            loss_value = info.get("value")
            loss_str = f"{loss_value:.6f}"
            self.batch_log.append((epoch, batch_idx, loss_value))

        # Log periodically
        if batch_idx % self.log_every == 0:
            log.info(f"[BatchLossLogger] Epoch {epoch} Batch {batch_idx} {loss_str}")

        # Optionally write to CSV
        if self.csv_path:
            import csv
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    if trainer.__class__.__name__ == "AdversarialTrainer":
                        writer.writerow(["epoch", "batch_idx", "loss_g", "loss_d"])
                    else:
                        writer.writerow(["epoch", "batch_idx", "loss"])
                if trainer.__class__.__name__ == "AdversarialTrainer":
                    writer.writerow([epoch, batch_idx, loss_g, loss_d])
                else:
                    writer.writerow([epoch, batch_idx, loss_value])



class EpochLossCSVLogger(Callback):
    """Logs epoch train and validation losses for both Trainer and AdversarialTrainer."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_loss_file = os.path.join(self.output_dir, "train_loss.csv")
        self.val_loss_file = os.path.join(self.output_dir, "val_loss.csv")

        # Create CSV headers if missing
        if not os.path.exists(self.train_loss_file):
            with open(self.train_loss_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss_g", "train_loss_d", "train_loss"])

        if not os.path.exists(self.val_loss_file):
            with open(self.val_loss_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_loss"])

    def on_epoch_end(self, trainer):
        """Append epoch-level training losses."""
        with open(self.train_loss_file, "a", newline="") as f:
            writer = csv.writer(f)
            if trainer.__class__.__name__ == "AdversarialTrainer":
                writer.writerow(
                    [
                        trainer.current_epoch,
                        getattr(trainer, "last_train_loss_g", None),
                        getattr(trainer, "last_train_loss_d", None),
                        None,
                    ]
                )
                log.info(
                    f"[EpochLossCSVLogger] Epoch {trainer.current_epoch} "
                    f"G={trainer.last_train_loss_g:.4f}, D={trainer.last_train_loss_d:.4f}"
                )
            else:
                writer.writerow(
                    [
                        trainer.current_epoch,
                        None,
                        None,
                        getattr(trainer, "last_train_loss", None),
                    ]
                )
                log.info(
                    f"[EpochLossCSVLogger] Epoch {trainer.current_epoch} "
                    f"train_loss={trainer.last_train_loss:.4f}"
                )

    def on_validation_end(self, trainer):
        """Append validation loss."""
        with open(self.val_loss_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trainer.current_epoch, getattr(trainer, "val_loss", None)])
        log.info(
            f"[EpochLossCSVLogger] Logged val loss for epoch {trainer.current_epoch}"
        )


class Timer(Callback):
    import time
    def on_epoch_begin(self, trainer):
        """Record start time for the epoch."""
        self.start_time = self.time.time()

    def on_epoch_end(self, trainer):
        """Compute and print epoch duration."""
        duration = self.time.time() - self.start_time
        log.info(f"[Timer] Epoch {trainer.current_epoch} duration: {duration:.2f}s")


class GradientClippingLogger(Callback):
    def on_batch_end(self, trainer, loss):
        """Compute and print the gradient norm after clipping.

        Note: this is a monitoring callback and does not perform clipping
        itself. It assumes gradients are available on trainer.model.
        """
        total_norm = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        log.info(
            f"[GradientClippingLogger] Gradient norm after clipping: {total_norm:.4f}"
        )


class ImageCallback(Callback):
    def __init__(self, output_dir, every_n_epochs=5, n_samples=16, use_dict= False):
        self.output_dir = os.path.join(output_dir, "samples")
        os.makedirs(self.output_dir, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.use_dict = use_dict

    def on_epoch_end(self, trainer):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        trainer.model.eval()
        with torch.no_grad():
            if self.use_dict:
                samples = trainer.model.sample_train(self.n_samples, batch = trainer.current_batch)
            else:
                samples = trainer.model.sample_train(self.n_samples)
            save_image(
                samples,
                os.path.join(
                    self.output_dir, f"epoch_{trainer.current_epoch + 1:04d}.png"
                ),
                normalize=True,
                value_range=(-1, 1),
                nrow=int(self.n_samples**0.5),
            )
        trainer.model.train()
        print(
            f"[ImageCallback] Saved generated samples for epoch {trainer.current_epoch + 1}"
        )



log = logging.getLogger(__name__)

class MemoryMonitorCallback(Callback):
    """
    Logs GPU and CPU memory usage at key stages (epoch begin, epoch end, or periodically).
    Works with the Trainer's callback system.
    """

    def __init__(self, interval: int = 50):
        """
        Parameters
        ----------
        interval : int
            Log memory every N batches (default=50). Use 0 to disable batch logging.
        """
        self.interval = interval
        self.process = psutil.Process()


    def on_train_begin(self, trainer):
        log.info("[Memory] Monitoring started")

    def on_epoch_begin(self, trainer):
        self._log_memory(f"[Epoch {trainer.current_epoch}] Begin")

    def on_batch_end(self, trainer):
        if self.interval > 0 and (trainer.loss_batch["batch_idx"] + 1) % self.interval == 0:
            self._log_memory(f"[Epoch {trainer.current_epoch}] Batch {trainer.loss_batch['batch_idx'] + 1}")

    def on_validation_end(self, trainer):
        self._log_memory(f"[Epoch {trainer.current_epoch}] After Validation")

    def on_epoch_end(self, trainer):
        self._log_memory(f"[Epoch {trainer.current_epoch}] End")

    def on_train_end(self, trainer):
        log.info("[Memory] Monitoring stopped")


    def _log_memory(self, prefix: str):
        # GPU memory
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            res = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
        else:
            alloc = res = peak = 0.0

        # CPU memory
        cpu_mem = self.process.memory_info().rss / 1024**3

        log.info(
            f"{prefix} | [CPU] {cpu_mem:.2f} GB | [GPU] alloc={alloc:.2f} GB, res={res:.2f} GB, peak={peak:.2f} GB"
        )
