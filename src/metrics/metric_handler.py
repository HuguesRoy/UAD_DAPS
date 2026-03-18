import pandas as pd
import math
import logging
import numpy as np

log = logging.getLogger(__name__)


class MetricHandler:
    """
    Modular handler for computing, aggregating, and exporting evaluation metrics.

    This class manages multiple metric objects and provides a unified interface
    to compute per-sample metrics during testing or validation. It supports both
    batch-wise updates and dataset-level aggregation.

    Parameters
    ----------
    metrics : list
        List of metric objects. Each metric must define:
          - `name` (str): unique metric name.
          - `prediction` (str): key in the model's output_dict corresponding to predictions.
          - `target` (str): key in the model's output_dict corresponding to ground truth.
          - `__call__(pred, target)`: callable that computes the metric value.

    Attributes
    ----------
    metric_objects : list
        The list of metric objects provided at initialization.
    metrics : dict
        Dictionary mapping each metric name -> list of computed values.
    per_sample_results : list of dict
        Stores per-sample metrics (and optional dataset indices) for CSV export.

    Methods
    -------
    reset()
        Clears all stored results between evaluation runs.
    update(output_dict, sample_indices=None)
        Computes metrics for each sample in the batch and stores results.
    compute()
        Returns mean metric values aggregated across all samples.
    save_csv(path)
        Saves all per-sample metrics to a CSV file for later analysis.

    Example
    -------
    >>> handler = MetricHandler([DiceMetric(), MAEMetric()])
    >>> handler.update(output_dict, sample_indices=[0, 1, 2])
    >>> results = handler.compute()
    >>> handler.save_csv("results/per_sample_metrics.csv")
    """
    
    def __init__(self, metrics):
        self.metric_objects = metrics
        self.metrics = {metric.name: [] for metric in metrics}
        self.per_sample_results = []  # For saving to CSV

    def reset(self):
        for name in self.metrics:
            self.metrics[name] = []
        self.per_sample_results = []

    def update(self, output_dict, sample_indices=None):
        batch_size = output_dict[self.metric_objects[0].prediction].shape[0]
        for i in range(batch_size):
            result = {}
            for metric in self.metric_objects:
                pred = output_dict[metric.prediction][i]
                target = output_dict[metric.target][i]
                try:
                    value = metric(pred, target)
                except SystemExit as e:
                    log.error(f"SystemExit raised inside metric {metric.name}: code={e.code}")
                    raise
                except Exception as e:
                    log.exception(f"Metric {metric.name} failed on sample {i}")
                    raise

                self.metrics[metric.name].append(value)
                result[metric.name] = value
            
            if sample_indices is not None:
                result["index"] = sample_indices[i]
            self.per_sample_results.append(result)

    def compute(self):
        results = {}
        for name, values in self.metrics.items():
            if not values:
                results[name] = {"mean": None, "std": None}
            else:
                mean = np.nanmean(values)
                std = np.nanstd(values)
                results[name] = {"mean": mean, "std": std}
        return results

    def save_csv(self, path):
        df = pd.DataFrame(self.per_sample_results)
        df.to_csv(path, index=False)
