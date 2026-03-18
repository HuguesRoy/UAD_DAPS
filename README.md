# UAD_DAPS

This repository implements **UAD-DAPS**, a diffusion-based framework for unsupervised anomaly detection.

The project is fully **Hydra-based**: all components (model, data, training, testing) are defined via configuration files.

---

## 1. Setup

```bash
conda create -n uad_daps python=3.10
conda activate uad_daps

pip install torch torchvision torchaudio
pip install hydra-core omegaconf
pip install clinicadl
```

Set project root (required):

```bash
export PROJECT_ROOT=$(pwd)
```

---

## 2. Training

Training is launched with:

```bash
HYDRA_FULL_ERROR=1 python src/train.py \
  --config-path $CONFIG_PATH \
  --config-name $CONFIG_NAME
```

### Key config components

Training config typically includes:

- model: `sde_edm`
- dataset: e.g. `sanity_brats`
- dataloader
- optimizer: `adamW`
- trainer
- wrapper
- EMA
- callbacks

Example parameters:

```yaml
trainer:
  train_config:
    n_epoch: 1500
    clip_grad_norm: 1.0

dataloader:
  train_loader:
    batch_size: 32
```

### Output

Each run creates a folder containing:

- checkpoints (`.pt`)
- logs
- `.hydra/config.yaml` (full resolved config)

---

## 3. Testing / Evaluation

Testing is done using a **predictor + pretrained checkpoint**.

Launch:

```bash
HYDRA_FULL_ERROR=1 python src/run_experiment.py \
  --config-path $CONFIG_PATH \
  --config-name $CONFIG_NAME
```

---

### Required inputs

- trained model config:
  ```
  /path/to/run/.hydra/config.yaml
  ```
- checkpoint:
  ```
  /path/to/run/checkpoints/epoch_xxx.pt
  ```

---

### Predictor setup

Example:

```yaml
n_models: 1
predictor: predictor_test_brats

model1:
  config: /path/to/.hydra/config.yaml
  ckpt:   /path/to/checkpoints/epoch_xxx.pt
```

---

### Output

- metrics (e.g. AP, Dice)
- saved predictions (optional)
- results stored in `hydra.run.dir`

---

## 4. Workflow

### Train
```bash
python src/train.py ...
```

→ produces:
- checkpoint
- config

### Test
```bash
python src/run_experiment.py ...
```

→ requires:
- checkpoint
- config

---

## 5. Notes

- Paths in configs must be adapted
- Hydra handles all instantiation
- Designed for reproducible experiments via config composition

---

## 6. Minimal commands

```bash
# Train
python src/train.py --config-path ... --config-name ...

# Test
python src/run_experiment.py --config-path ... --config-name ...
```
