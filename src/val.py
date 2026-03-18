"""Validation entrypoint using Hydra configuration.

This module exposes a `main(cfg: DictConfig)` function decorated with
`hydra.main`. It performs validation for reconstruction-based anomaly detection
models, supporting both hyperparameter tuning and reconstruction statistics
estimation (for Z-score normalization).
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


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Main Hydra validation entry point."""

    log.info("Hydra logger initialized")
    log.info("Hydra config loaded")
    print(OmegaConf.to_yaml(cfg, resolve=True))


    val_dataset = instantiate(cfg.data.data_val) if "data_val" in cfg.data else None
    if val_dataset is None:
        log.warning("[Validator] No validation dataset found — exiting.")
        return
    log.info("Validation dataset loaded")

    val_loader = instantiate(cfg.dataloader.val_loader)(dataset=val_dataset)
    log.info("Validation dataloader instantiated")


    wrapper = instantiate(cfg.wrapper)
    log.info("Data wrapper loaded")



    model = instantiate(cfg.model)

    predictor = instantiate(cfg.predictor)(model = model)
    
    log.info(f"Predictor instantiated: {type(predictor).__name__}")

    metrics = (
        [instantiate(m_cfg) for m_cfg in cfg.metrics.values()]
        if "metrics" in cfg
        else []
    )
    log.info(f"Metrics Loaded: {[m.name for m in metrics] if metrics else 'None'}")


    validator_partial = instantiate(cfg.validator)
    validator = validator_partial(
        predictor=predictor,
        data_wrapper=wrapper,
        device=cfg.device,
        metrics=metrics
    )
    
    log.info(f"Validator instantiated: {type(validator).__name__}")

    log.info("Starting validation process...")

    best_params = None
    best_score = None

    if getattr(cfg.validator, "tune_params", True):
        if "param_grid" in cfg:
            select_metric = getattr(cfg, "select_metric", (metrics[0].name if metrics else "mse"))
            mode = getattr(cfg, "sweep_mode", "grid")

            best_params, best_score = validator.tune_params(
                val_loader,
                param_grid=cfg.param_grid,
                select_metric=select_metric,
                mode=mode,
            )

            log.info(
                f"[Validator] Best validation parameters: {best_params}, score={best_score:.6f}"
            )
        else:
            log.warning(
                "[Validator] tune_params=True but no param_grid provided - skipping tuning."
            )
    else:
        log.info("[Validator] Skipping hyperparameter tuning.")

    if getattr(cfg.validator, "compute_stats", True):
        stats = validator.compute_reconstruction_stats(val_loader, params=best_params)
        if "metrics" in stats:
            m = getattr(cfg, "select_metric", (metrics[0].name if metrics else "mse"))
            log.info(
                f"[Validator] Stats ({m}) — mean={stats['metrics'][m]['mean']:.6f}, std={stats['metrics'][m]['std']:.6f}"
            )
        else:
            log.info(f"[Validator] Stats — {stats}")

    else:
        log.info("[Validator] Skipping reconstruction statistics computation.")


if __name__ == "__main__":
    main()
