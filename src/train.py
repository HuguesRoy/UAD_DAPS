"""Training entrypoint using Hydra configuration.

This module exposes a `main(cfg: DictConfig)` function decorated with
`hydra.main`. It supports both standard and adversarial training workflows,
depending on the configured `trainer` target.

Responsibilities:
    - Load datasets
    - Instantiate model(s), optimizers, EMA, callbacks, and wrapper
    - Launch Trainer or AdversarialTrainer depending on cfg.trainer._target_
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path
import logging
import torch
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

log = logging.getLogger(__name__)

print("torch:", torch.__version__)
print("cuda version in torch:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())

def load_callbacks(callbacks_list, cfg):
    """
    Load callback configs from configs/callbacks/<name>.yaml and instantiate them.

    - callbacks_list: list[str] like ["model_checkpoint", "logger", "sample_save"]
    - cfg: the already-composed Hydra cfg (DictConfig)
    """
    # Where your repo configs live (works under submitit because it uses original cwd)
    config_root = Path(hydra.utils.get_original_cwd()) / "configs" / "callbacks"

    callback_objects = []

    for cb_name in callbacks_list:
        path = config_root / f"{cb_name}.yaml"
        log.info(f"[Callbacks] Loading {path}")

        if not path.exists():
            raise FileNotFoundError(f"Callback config not found: {path}")

        cb_cfg = OmegaConf.load(path)

        # Your callback yamls look like:
        # checkpoint:
        #   _target_: ...
        #   output_dir: ${trainer.output_dir}
        #
        # so unwrap the first key ("checkpoint", "logger", ...)
        cb_subcfg = list(cb_cfg.values())[0]
        if len(cb_subcfg) == 1 and "_target_" not in cb_subcfg:
            cb_subcfg = list(cb_subcfg.values())[0]

        # Resolve ${trainer.output_dir} etc against the global cfg
        # This merges under a root so interpolations like ${trainer.output_dir} work.
        merged = OmegaConf.merge(cfg, {"__cb__": cb_subcfg})
        cb_resolved = merged["__cb__"]

        cb_instance = instantiate(cb_resolved)
        callback_objects.append(cb_instance)

    return callback_objects



@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Main Hydra training entry point."""

    log.info("Hydra logger initialized")
    log.info("Hydra config loaded")
    print(OmegaConf.to_yaml(cfg, resolve=True))


    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    train_dataset = instantiate(cfg.data.data_train)
    val_dataset = instantiate(cfg.data.data_val) if "data_val" in cfg.data else None
    log.info("Datasets loaded successfully")

    # -------------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------------
    partial_train_loader = instantiate(cfg.dataloader.train_loader)
    train_loader = partial_train_loader(dataset=train_dataset)
    val_loader = (
        instantiate(cfg.dataloader.val_loader)(dataset=val_dataset)
        if val_dataset is not None
        else None
    )
    log.info("Dataloaders instantiated")

    # -------------------------------------------------------------------------
    # Callbacks, EMA, Wrapper
    # -------------------------------------------------------------------------
    
    callbacks = []
    if "callbacks" in cfg and cfg.callbacks is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]

    log.info(f"[Hydra] Loaded callbacks: {[type(cb).__name__ for cb in callbacks]}")


    wrapper = instantiate(cfg.wrapper)
    ema = instantiate(cfg.ema) if "ema" in cfg else None
    log.info("Callbacks, Wrapper, and EMA loaded")

    # -------------------------------------------------------------------------
    # Choose trainer type
    # -------------------------------------------------------------------------
    trainer_target = cfg.trainer._target_

    if "AdversarialTrainer" in trainer_target:
        log.info("Initializing Adversarial training setup")

        # Instantiate generator and discriminator separately
        model = instantiate(cfg.model)
        log.info(
            f"Loaded Generator: {cfg.model.generator._target_} | "
            f"Discriminator: {cfg.model.discriminator._target_}"
        )

        # Optimizers
        opt_gen = instantiate(cfg.optimizer_generator)(model.generator.parameters())
        opt_disc = instantiate(cfg.optimizer_discriminator)(
            model.discriminator.parameters()
        )

        # Trainer
        trainer_partial = instantiate(cfg.trainer)
        trainer = trainer_partial(
            model=model,
            optim_g=opt_gen,
            optim_d=opt_disc,
            data_wrapper=wrapper,
            output_dir=cfg.trainer.output_dir,
            callbacks=callbacks,
            ema=ema,
        )

    else:
        log.info("Initializing Standard training setup")

        # Standard single-model setup
        model = instantiate(cfg.model)
        optimizer = instantiate(cfg.optimizer)(model.parameters())

        trainer_partial = instantiate(cfg.trainer)
        trainer = trainer_partial(
            model=model,
            optimizer=optimizer,
            data_wrapper=wrapper,
            callbacks=callbacks,
            ema=ema,
            output_dir=cfg.trainer.output_dir,
        )

    log.info("Launching training process")
    trainer.train(train_loader=train_loader, validation_loader=val_loader)


if __name__ == "__main__":
    main()
