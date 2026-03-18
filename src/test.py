"""Testing entrypoint using Hydra configuration.

This module provides a `main(cfg: DictConfig)` function (decorated with
`hydra.main`) which sets up the test dataset and loader, loads the model
(optionally from a checkpoint), prepares metrics and saver/handlers and
runs the `Tester`.

The configuration is expected to contain `data.data_test`,
`dataloader.test_loader` and `model`. Optional sections include
`resume`, `wrapper`, `metrics`, `saver` and `trainer`.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
from pathlib import Path
import logging
import torch

# add src/ to PYTHONPATH programmatically
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

log = logging.getLogger(__name__)


@hydra.main(
    config_path=None,  # <-- allow external config folders
    config_name="test",  # <-- name of config in AD_hypo_30
    version_base=None,
)
def main(cfg: DictConfig):
    """Hydra entry point for running a test/evaluation.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing test and model settings.
    """
    print(OmegaConf.to_yaml(cfg))
    log.info("Hydra logger initialized")
    log.info("Hydra config loaded")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------------------
    # Dataset
    # ------------------------------
    test_dataset = instantiate(cfg.data.data_test)
    log.info("Test dataset loaded")

    # Dataloader
    partial_loader_test = instantiate(cfg.dataloader.test_loader)
    test_loader = partial_loader_test(dataset=test_dataset)
    log.info("Test dataloader loaded")

    # ------------------------------
    # Model loading
    # ------------------------------
    
    predictor = instantiate(cfg.predictor)
    log.info(f"Predictor Loaded: {cfg.predictor._target_}")

    # ------------------------------
    # Wrapper
    # ------------------------------
    wrapper = instantiate(cfg.wrapper)
    log.info("Wrapper Loaded")

    # ------------------------------
    # Handlers
    # ------------------------------
    metrics = (
        [instantiate(m_cfg) for m_cfg in cfg.metrics.values()]
        if "metrics" in cfg
        else []
    )
    saver = instantiate(cfg.saver) if "saver" in cfg else None

    log.info(f"Metrics Loaded: {[m.name for m in metrics] if metrics else 'None'}")

    # ------------------------------
    # Tester
    # ------------------------------
    test_partial = instantiate(cfg.tester)

    tester = test_partial(
        predictor=predictor,
        data_wrapper=wrapper,
        metrics=metrics,
        saver=saver,
    )
    log.info("Tester Loaded")

    # ------------------------------
    # Launch Testing
    # ------------------------------
    log.info("Launching testing")
    tester.test(test_loader)
    log.info("Testing complete")


if __name__ == "__main__":
    main()
