import torch
from sklearn.metrics import average_precision_score
from multiprocessing import Pool
import numpy as np
from functools import partial
from .utils_ssim import ssim

import logging

log = logging.getLogger(__name__)

def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError("Predictions must be binary")
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError("Targets must be binary")

    # Compute Dice
    pred_sum = preds.sum()
    target_sum = targets.sum()

    if pred_sum == 0 and target_sum == 0:
        return np.nan

    if pred_sum == 0 or target_sum == 0:
        return np.nan

    # Case 3: normal Dice
    intersection = preds[targets == 1].sum()
    return 2.0 * intersection / (pred_sum + target_sum)
    
def _dice_multiprocessing(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    return compute_dice(np.where(preds > threshold, 1, 0), targets)


class AveragePrecision:
    def __init__(self,in_mask=False):
        self.name = "average_precision"
        self.prediction = "anomaly_map"
        self.target = "seg_mask"

        if in_mask:
            self.name = "average_precision_in_brain_mask"
            self.prediction = "anomaly_map_in_brain_mask"
            self.target = "seg_mask_in_brain_mask"


    def __call__(self, pred, target):
        # Convert torch -> numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # Flatten
        pred = pred.ravel()
        target = target.ravel()

        # Remove NaN / inf
        valid_mask = np.isfinite(pred) & np.isfinite(target)
        pred = pred[valid_mask]
        target = target[valid_mask]

        if np.sum(target) == 0:
            return np.nan

        ap = average_precision_score(target, pred)

        if np.isnan(ap):
            ap = 0.0

        return float(ap)


class BestDiceIntensity:
    def __init__(
        self, n_threshold=100, num_processes=4,in_mask=False
    ):
        self.name = "best_dice_intensity"
        self.prediction = "anomaly_map"
        self.target = "seg_mask"

        if in_mask:
            self.name = "best_dice_intensity_in_brain_mask"
            self.prediction = "anomaly_map_in_brain_mask"
            self.target = "seg_mask_in_brain_mask"

        self.n_threshold = n_threshold
        self.num_processes = num_processes

    def __call__(self, pred, target):
        if isinstance(pred, torch.Tensor):
            preds = pred.detach().cpu().numpy()
        else:
            preds = pred
        if isinstance(target, torch.Tensor):
            targets = target.detach().cpu().numpy()
        else:
            targets = target

        if preds.dtype == np.bool_:
            preds = preds.astype(np.float32)
        if targets.dtype == np.bool_:
            targets = targets.astype(np.float32)

        thresholds = np.linspace(preds.min(), preds.max(), self.n_threshold)

        with Pool(self.num_processes) as pool:
            fn = partial(_dice_multiprocessing, preds, targets)
            scores = pool.map(fn, thresholds)

        scores = np.asarray(scores, dtype=float)

        # If everything is NaN, define Dice = 0.0 for this sample
        if np.all(np.isnan(scores)):
            return 0.0

        return np.nanmax(scores)



class DiceScore:
    def __init__(self):
        self.name = "dice"
        self.prediction = "anomaly_seg"
        self.target = "seg_mask"

    def __call__(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if pred.dtype == np.bool_:
            pred = pred.astype(np.float32)

        intersection = np.sum(pred[target == 1])
        den = np.sum(pred) + np.sum(target)

        if den == 0:
            dice = 1.0   # both empty
        else:
            dice = 2 * intersection / den

        return dice


class MSE():

    def __init__(self, mask_extension=""):

        self.name = mask_extension + "mse"
        self.prediction = mask_extension +"reconstruction"
        self.target = mask_extension +"x"

    def __call__(self, pred, target):

        return torch.mean((pred - target) ** 2).item()
    
    
class SSIM():

    def __init__(self, mask_extension=""):

        self.name = mask_extension + "ssim"
        self.prediction = mask_extension + "reconstruction"
        self.target = mask_extension + "x"

    def __call__(self, pred, target):
        # add batch dimension
        return ssim(pred.unsqueeze(0),target.unsqueeze(0)).item()
    