import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import csv
import itertools
from metrics.metric_handler import MetricHandler
from typing import Dict, List, Any, Iterable, Tuple, Optional
import re
from pathlib import Path

log = logging.getLogger(__name__)

class ReconMSE:
    name = "mse"
    prediction = "reconstruction"
    target = "x"

    def __call__(self, pred, target):
        # pred and target are single-sample tensors here (MetricHandler slices [i])
        return ((pred - target) ** 2).mean().item()



class Validator:
    """
    Validator for unsupervised anomaly detection models (reconstruction-based).

    It supports two main tasks:
      1. Hyperparameter tuning via validation reconstruction performance.
      2. Estimation of reconstruction error statistics (mean/std) for Z-score calibration.

    Parameters
    ----------
    predictor : nn.Module or callable
        Model or predictor implementing `predict(batch_dict, **params)` or
        `reconstruct(batch_dict, **params)`. It must return reconstructed images or tensors.
    data_wrapper : callable
        Function mapping dataset batches to standardized dicts:
        { "image", "segmentation", "mask", "label" }.
    device : str
        Device for evaluation (e.g. "cuda" or "cpu").
    output_dir : str, optional
        Directory to save results and validation statistics.
    metric : callable, optional
        Function to compute reconstruction error. Defaults to L2.
    """

    def __init__(
        self, predictor, data_wrapper, device="cuda", output_dir=None, metrics=None
    ):
        print("[DEBUG] Validator initialized with:")
        print("   predictor:", type(predictor))
        print("   data_wrapper:", type(data_wrapper))
        print("   device:", device)

        self.predictor = predictor

        if hasattr(self.predictor, "to"):
            self.predictor.to(device)
        elif hasattr(self.predictor, "model") and hasattr(self.predictor.model, "to"):
            self.predictor.model.to(device)

        self.data_wrapper = data_wrapper
        self.device = device
        self.output_dir = output_dir
        
        self.metrics = metrics if metrics else [ReconMSE()]
        self.metric_handler = MetricHandler(self.metrics)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.predictor is None:
            raise ValueError(
                "[Validator] Predictor was not provided - did you forget to call validator_partial(...) with predictor=?"
            )


    def default_metric(self, dict_pred):
        """Default L2 reconstruction error."""
        x = dict_pred["x"]
        x_rec = dict_pred["reconstruction"]
        return ((x - x_rec) ** 2).mean(dim=tuple(range(1, x.ndim)))

    def _sanitize_value(self, v):
        # Make values safe for filenames while keeping them readable.
        # Examples: 0.1 -> "0.1", "foo/bar" -> "foo-bar", (1,2) -> "(1,2)"
        s = str(v)
        s = s.strip()
        s = s.replace(" ", "")
        s = s.replace("/", "-")
        s = s.replace("\\", "-")
        s = re.sub(r"[^A-Za-z0-9_.=\-(),+]", "-", s)  # conservative safe set
        return s


    def _params_to_slug(self, params: dict | None) -> str:
        if not params:
            return "default"
        parts = []
        for k in sorted(params.keys()):
            parts.append(f"{k}={self._sanitize_value(params[k])}")
        return "__".join(parts)


    def _iter_param_combinations(self, param_grid: dict, mode: str = "grid"):
        if not param_grid:
            yield {}
            return

        names = list(param_grid.keys())
        lists = [param_grid[k] for k in names]

        if mode == "grid":
            for values in itertools.product(*lists):
                yield dict(zip(names, values))

        elif mode == "coupled":
            lengths = [len(v) for v in lists]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"[Validator] coupled mode requires equal lengths, got {dict(zip(names, lengths))}"
                )
            for i in range(lengths[0]):
                yield {names[j]: lists[j][i] for j in range(len(names))}

        else:
            raise ValueError(f"[Validator] Unknown mode: {mode}")

    @torch.no_grad()
    def tune_params(self, val_loader, param_grid: dict, select_metric: str, mode: str = "grid", csv_name: str = "sweep_stats.csv"):
        """
        Evaluate all param combinations and save per-combo stats/errors + a CSV summary.

        mode="grid": cartesian product (2D/3D/ND)
        mode="coupled": zip lists; asserts equal lengths
        """
        best_score = float("inf")
        best_params = None
        rows = []

        # Where CSV goes
        csv_path = None
        if self.output_dir:
            out_dir = Path(self.output_dir) / "param_sweep"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / csv_name

        for params in self._iter_param_combinations(param_grid, mode=mode):
            stats = self.compute_reconstruction_stats(val_loader, params)
            if select_metric not in stats["metrics"]:
                raise KeyError(f"[Validator] select_metric='{select_metric}' not found. Available: {list(stats['metrics'].keys())}")

            metric_mean = stats["metrics"][select_metric]["mean"]

            row = {**params}
            for name, values in stats["metrics"].items():
                row[f"{name}_mean"] = values["mean"]
                row[f"{name}_std"] = values["std"]

            rows.append(row)

            if metric_mean < best_score:
                best_score = metric_mean
                best_params = params


        # Write CSV
        if csv_path and rows:
            # stable param order
            param_cols = sorted(param_grid.keys())

            # metric cols discovered from first row
            metric_cols = [k for k in rows[0].keys() if k not in param_cols]

            fieldnames = param_cols + metric_cols

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})


        log.info(f"[Validator] Best params: {best_params}, score={best_score:.6f}")
        return best_params, best_score


    @torch.no_grad()
    def compute_reconstruction_stats(self, val_loader, params: dict | None = None):

        if hasattr(self.predictor, "model"):
            self.predictor.model.eval()
        elif hasattr(self.predictor, "eval"):
            self.predictor.eval()

        if params:
            self.predictor.set_params(**params)

        slug = self._params_to_slug(params)
        out_dir = None
        if self.output_dir:
            out_dir = Path(self.output_dir) / "param_sweep"
            out_dir.mkdir(parents=True, exist_ok=True)

        self.metric_handler.reset()

        for batch in tqdm(val_loader, desc="[Validator] Evaluating"):
            batch = self.data_wrapper(batch)

            output_dict = self.predictor.predict(batch)

            # ensure ground truth is available to metrics
            if "x" not in output_dict and "image" in batch:
                output_dict["x"] = batch["image"]

            self.metric_handler.update(output_dict)

        results = self.metric_handler.compute()

        # Save per-sample CSV
        if out_dir:
            self.metric_handler.save_csv(out_dir / f"per_sample__{slug}.csv")
            torch.save(
                {"params": params or {}, "metrics": results},
                out_dir / f"stats__{slug}.pt",
            )

        return {"params": params or {}, "metrics": results}


