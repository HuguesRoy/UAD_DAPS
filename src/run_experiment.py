import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import os
from omegaconf import open_dict
import sys
from pathlib import Path
import logging
# add src/ to PYTHONPATH programmatically
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

log = logging.getLogger(__name__)


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return OmegaConf.load(path)

def try_assign(cfg_node, field, value):
    if field in cfg_node:
        cfg_node[field] = value

        
@hydra.main(version_base=None)
def main(cfg):
    print("\n========================")
    print("  BASE CONFIG LOADED")
    print("========================")
    print(OmegaConf.to_yaml(cfg))

    log.info(f"H Hydra run dir: {hydra.core.hydra_config.HydraConfig.get().run.dir}")
    
    # -----------------------------------------------------------
    # 1) Resolve model family predictor folder
    # -----------------------------------------------------------
    model_cfg_dir = cfg.job.searchpath.replace("file://", "")
    predictor_name = cfg.job.predictor  # e.g. predictor_test
    predictor_path = os.path.join(model_cfg_dir, "predictor", predictor_name + ".yaml")

    print(f"\n[INFO] Loading predictor config:   {predictor_path}")
    predictor_cfg = load_yaml(predictor_path)

    print("\n==============================")
    print("  PREDICTOR CONFIG LOADED")
    print("==============================")
    print(OmegaConf.to_yaml(predictor_cfg))

    # -----------------------------------------------------------
    # 2) Inject model weights depending on n_models
    # -----------------------------------------------------------
    n = cfg.job.n_models

    if n == 1:
        predictor_cfg.model.cfg_path = cfg.job.model1.config
        predictor_cfg.model.checkpoint_path = cfg.job.model1.ckpt
        predictor_cfg.model.device = cfg.device

    elif n == 2:

        if "models" not in predictor_cfg:
            raise ValueError(
                f"Your predictor config must define a 'models' section. "
                f"Found keys: {list(predictor_cfg.keys())}"
            )

        model_keys = list(predictor_cfg.models.keys())
        expected = cfg.job.n_models

        if len(model_keys) != expected:
            raise ValueError(
                f"Predictor expects {expected} models, "
                f"but predictor config defines: {model_keys}"
            )

        # Map model1, model2, ... dynamically
        job_models = [cfg.job.model1, cfg.job.model2][:expected]

        for key, job_model in zip(model_keys, job_models):
            m_cfg = predictor_cfg.models[key]
            m_cfg.cfg_path = job_model.config
            m_cfg.checkpoint_path = job_model.ckpt
            m_cfg.device = cfg.device

        print("\n==============================")
        print("  MODELS INJECTED INTO PREDICTOR CFG")
        print("==============================")
        print(OmegaConf.to_yaml(predictor_cfg.models))


    else:
        raise ValueError(f"Unsupported n_models={n}")

    # -----------------------------------------------------------
    # 3) Inject global device if needed
    # -----------------------------------------------------------
    if "test_config" in predictor_cfg:
        predictor_cfg.test_config.device = cfg.device
    elif "predictor_config" in predictor_cfg:
        predictor_cfg.predictor_config.device = cfg.device

    print("\n==============================")
    print("  FINAL MERGED PREDICTOR CFG")
    print("==============================")
    print(OmegaConf.to_yaml(predictor_cfg, resolve=True))

    # -----------------------------------------------------------
    # 4) Instantiate predictor
    # -----------------------------------------------------------
    predictor = instantiate(predictor_cfg)
    log.info(f"Predictor Loaded: {predictor_cfg._target_}")

    if cfg.job.data_override is not None:
        print("\n==============================")
        print("  APPLYING DATA OVERRIDE")
        print("==============================")

        with open_dict(cfg.data):
            cfg.data = OmegaConf.merge(cfg.data, cfg.job.data_override)

        print("[OK] Data override applied.")
        print(OmegaConf.to_yaml(cfg.data))
    
    test_dataset = instantiate(cfg.data.data_test)
    log.info("Test dataset loaded")

    # Dataloader
    partial_loader_test = instantiate(cfg.dataloader.test_loader)
    test_loader = partial_loader_test(dataset=test_dataset)
    log.info("Test dataloader loaded")

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

    return {"status": "ok"}


if __name__ == "__main__":
    main()
