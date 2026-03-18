import torch
from metrics.metric_handler import MetricHandler
import os
from typing import Optional
from tqdm import tqdm
import logging
import time

log = logging.getLogger(__name__)

class Tester:
    """
    Modular testing pipeline for evaluating a trained model.

    The `Tester` orchestrates model inference, metric evaluation, and output saving
    during testing or validation. It integrates seamlessly with modular components
    such as `MetricHandler` (for quantitative evaluation) and `SaverHandler`
    (for qualitative or tensor output storage).

    Parameters
    ----------
    predictor : torch.nn.Module
        The trained model or predictor to evaluate. Must implement a `predict(x, mask)`
        method that returns a tuple of tensors.
    test_config : object
        Configuration object containing testing parameters. Must include the `device` attribute.
    data_wrapper : callable
        Wrapper class or function that prepares a batch dictionary for the predictor.
        Typically returns a dict with keys such as `"x"`, `"mask"`, `"seg"`, etc.
    output_dir : str
        Directory where metrics and results will be saved.
    metrics : list, optional
        List of metric objects passed to the `MetricHandler`. If `None`, no metrics are computed.
    saver : SaverHandler, optional
        Handler responsible for saving tensors or images. If `None`, no outputs are saved.

    Attributes
    ----------
    predictor : torch.nn.Module
        Model used for inference.
    metric_handler : MetricHandler or None
        Handles computation and aggregation of evaluation metrics.
    saver : SaverHandler or None
        Handles saving of model outputs during testing.
    output_results : str
        Directory where outputs (e.g. metrics, tensors) are stored.
    data_wrapper : callable
        Function or class that formats input batches for the predictor.
    device : str
        Device on which inference is performed (e.g., `'cuda'` or `'cpu'`).

    Methods
    -------
    test(test_loader)
        Runs the complete testing loop: prediction, metric computation, and optional saving.

    Example
    -------
    >>> tester = Tester(
    ...     predictor=model,
    ...     test_config=cfg.test,
    ...     data_wrapper=ClinicaDLWrapper,
    ...     output_dir="results",
    ...     metrics=[DiceMetric(), MAEMetric()],
    ...     saver=SaverHandler(output_dir="results/tensors", save_indices=[0, 2, 18])
    ... )
    >>> tester.test(test_loader)
    """
    def __init__(
        self,
        predictor,
        test_config,
        data_wrapper,
        output_dir,
        metrics = None,
        saver = None,
    ):
        self._init_config(test_config)
        self.predictor = predictor
        self.predictor.to(self.device)
        self.metric_handler = MetricHandler(metrics) if metrics else None
        self.saver = saver
        self.output_results = output_dir
        self.data_wrapper = data_wrapper

    def _init_config(self, config):
        self.device = config.device

    @torch.no_grad()
    def test(self, test_loader):
        log.info(f"Test loader length: {len(test_loader)}")

        if self.metric_handler:
            self.metric_handler.reset()
        
        total_infer_seconds = 0
        total_samples = 0


        for batch_idx, batch in enumerate(tqdm(test_loader)):
            dict_ord = self.data_wrapper(batch)

            bs = dict_ord["image"].shape[0]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            output_dict = self.predictor.predict(dict_ord)

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            total_infer_seconds += dt
            total_samples += bs

            # optional: store per-sample time for this batch
            output_dict["inference_time_per_sample"] = dt / bs

            log.debug("[DEBUG] Exiting predictor")
            output_dict["x"] = dict_ord["image"]
            output_dict["seg"] = dict_ord["seg"]
            output_dict["seg_mask"] = dict_ord["seg_mask"]

            sample_indices = dict_ord.get(
                "index",
                list(
                    range(
                        batch_idx * test_loader.batch_size,
                        (batch_idx + 1) * test_loader.batch_size,
                    )
                ),
            )

            if self.saver:
                self.saver.save_batch(
                    batch_idx=batch_idx,
                    batch_size=bs,
                    output_dict=output_dict,
                    sample_indices=sample_indices,
                )

            if self.metric_handler:
                self.metric_handler.update(output_dict, sample_indices=sample_indices)

        mean_infer_time_per_sample = total_infer_seconds / max(total_samples, 1)
        log.info(f"Mean inference time per sample: {mean_infer_time_per_sample:.6f} s "
                f"({1.0/mean_infer_time_per_sample:.2f} samples/s)")

        if self.metric_handler:
            results = self.metric_handler.compute()
            print("Test metrics:", results)
            self.metric_handler.save_csv(
                os.path.join(self.output_results, "per_sample_metrics.csv")
            )
