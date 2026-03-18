import os
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import Optional
from .ema import EMA

def load_model_from_config(
    cfg_path: str,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    key: str = "model",
    use_ema: bool = False,
):
    """
    Load a model from a Hydra config and (optionally) apply EMA weights.

    Args:
        cfg_path: Path to Hydra config (.yaml)
        device: Device for the instantiated model
        checkpoint_path: Optional path to checkpoint
        key: Key in the checkpoint for the model weights ("model" typically)
        use_ema: If True, load EMA weights and apply them to the model

    Returns:
        model: nn.Module
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Load YAML config into OmegaConf
    cfg = OmegaConf.load(cfg_path)

    # ----- Detect whether cfg is full training config or direct model config -----
    if "_target_" in cfg:
        model_cfg = cfg
        print("[load_model_from_config] Detected direct model config")
    elif "model" in cfg:
        model_cfg = cfg.model
        print("[load_model_from_config] Detected full training config")
    else:
        raise ValueError(
            f"Cannot find model definition in config. "
            f"Expected '_target_' or 'model'. Found keys: {list(cfg.keys())}"
        )

    # ----- Instantiate model -----
    model = instantiate(model_cfg).to(device)

    # ----- Load checkpoint -----
    if checkpoint_path is None:
        return model  # no weights

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch.serialization.add_safe_globals([ListConfig])

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If checkpoint is the full training dict, extract model weights
    if isinstance(ckpt, dict) and key in ckpt:
        state_dict = ckpt[key]
    else:
        state_dict = ckpt

    # ----- Load raw model weights -----
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[load_model_from_config] Missing keys: {missing}")
    if unexpected:
        print(f"[load_model_from_config] Unexpected keys: {unexpected}")

    # ----- Apply EMA weights if requested -----
    if use_ema:
        if "ema" not in ckpt:
            raise RuntimeError(
                f"use_ema=True but checkpoint has no EMA data: {checkpoint_path}"
            )

        ema_state = ckpt["ema"]

        ema = EMA(
            mu=ema_state.get("mu", 0.999),
            update_buffers=ema_state.get("update_buffers", True),
        )
        ema.load_state_dict(ema_state)
        ema.apply_to(model)

        print("[load_model_from_config] EMA weights applied.")

    return model
