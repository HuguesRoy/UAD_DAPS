import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
from pathlib import Path
import torch

# add src/ to PYTHONPATH programmatically
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print(ROOT)
@hydra.main(config_path=f"/{ROOT}/models/vqvae_spatial/configs", config_name="model/3D_vqvae_spatial", version_base=None)
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    model = instantiate(cfg.model)
    print(f"\n Instantiated model: {model.__class__.__name__}")
    model_dict = torch.load("/lustre/fswork/projects/rech/krk/commun/anomdetect/latent_DDPM/ADNI_PET/trained_models/vqvae_spatial/VQVAE_training_2026-01-23_17-22-51/final_model/model.pt", map_location=torch.device('cpu'), weights_only = True)
    model.load_state_dict(model_dict["model_state_dict"])

    encoder = model.encoder
    print(f"\n Instantiated model: {encoder.__class__.__name__}")
    print(f"\n Weights Loaded")

if __name__ == "__main__":
    print(ROOT)
    main()
