import os
import torch
import numpy as np
from typing import Dict, List, Optional


class SaverHandler:
    """
    Modular handler for saving model outputs during testing/validation.

    Modes:
      1. save_every = N: save all samples every N batches.
      2. save_indices = [i, j, ...]: save only these dataset indices.

    Options:
      - save_format: "pt", "npy", or "both"
      - per_sample: if True, always save one file per sample (even in save_every mode).
    """

    def __init__(
        self,
        output_dir: str,
        save_every: Optional[int] = None,
        save_indices: Optional[List[int]] = None,
        name_map: Optional[Dict[str, str]] = None,
        save_format: str = "pt",
        per_sample: bool = False,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if save_every is not None and save_indices:
            raise ValueError("`save_every` and `save_indices` are mutually exclusive.")

        if save_format not in {"pt", "npy", "both"}:
            raise ValueError("save_format must be one of {'pt', 'npy', 'both'}")

        self.save_every = save_every
        self.save_indices = save_indices or []
        self.name_map = name_map or {}
        self.save_format = save_format
        self.per_sample = per_sample

        self._saved = set()  # which indices are already saved (for save_indices mode)


    def _save_pt(self, tensor: torch.Tensor, path_root: str):
        """Save tensor as .pt (path_root without extension)."""
        path = path_root + ".pt"
        torch.save(tensor.detach().cpu(), path)

    def _save_npy(self, array: np.ndarray, path_root: str):
        """Save array as .npy (path_root without extension)."""
        path = path_root + ".npy"
        np.save(path, array)

    def _save_item(self, tensor_or_array, path_root: str):
        """
        Save one item (tensor, numpy array, or scalar) according to save_format.
        path_root: full path without extension.
        """
        if isinstance(tensor_or_array, np.ndarray):
            arr = tensor_or_array
            tensor = torch.from_numpy(arr)
        elif torch.is_tensor(tensor_or_array):
            tensor = tensor_or_array.detach().cpu()
            arr = tensor.numpy()
        else:
            tensor = torch.tensor(tensor_or_array).detach().cpu()
            arr = tensor.numpy()

        if self.save_format in {"pt", "both"}:
            self._save_pt(tensor, path_root)
        if self.save_format in {"npy", "both"}:
            self._save_npy(arr, path_root)


    # ---- public API ---- #

    def save_batch(
        self,
        batch_idx: int,
        batch_size: int,
        output_dict: Dict[str, torch.Tensor],
        sample_indices: List[int],
    ):
        """
        Parameters
        ----------
        batch_idx : int
            Index of the current batch (0-based).
        batch_size : int
            Batch size B.
        output_dict : dict
            E.g. {"x": tensor(B, ...), "x_rec": tensor(B, ...), ...}
        sample_indices : list[int]
            Global dataset indices for each sample in the batch.
        """

        # ---- Mode 1: save every N batches ---- #
        if self.save_every is not None:
            if batch_idx % self.save_every != 0:
                return

            if self.per_sample:
                for local_pos, global_idx in enumerate(sample_indices):
                    item_dir = os.path.join(self.output_dir, f"sample_{global_idx:05d}")
                    os.makedirs(item_dir, exist_ok=True)

                    for key, tensor in output_dict.items():
                        if tensor is None:
                            continue

                        if isinstance(tensor, np.ndarray):
                            sample_item = tensor[local_pos : local_pos + 1]

                        elif torch.is_tensor(tensor):
                            sample_item = tensor[local_pos].unsqueeze(0)

                        else:
                            if local_pos != 0:
                                continue
                            sample_item = tensor  # will be normalized in _save_item

                        base_name = self.name_map.get(key, key)
                        path_root = os.path.join(item_dir, base_name)
                        self._save_item(sample_item, path_root)

            else:
                batch_dir = os.path.join(self.output_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)

                for key, tensor in output_dict.items():
                    if tensor is None:
                        continue

                    base_name = self.name_map.get(key, key)
                    path_root = os.path.join(batch_dir, base_name)
                    self._save_item(tensor, path_root)

            return

        if self.save_indices:
            for local_pos, global_idx in enumerate(sample_indices):
                if global_idx in self.save_indices and global_idx not in self._saved:
                    self._saved.add(global_idx)
                    item_dir = os.path.join(self.output_dir, f"sample_{global_idx:05d}")
                    os.makedirs(item_dir, exist_ok=True)

                    for key, tensor in output_dict.items():
                        if tensor is None:
                            continue

                        if isinstance(tensor, np.ndarray):
                            sample_item = tensor[local_pos : local_pos + 1]
                        elif torch.is_tensor(tensor):
                            sample_item = tensor[local_pos].unsqueeze(0)
                        else:
                            sample_item = tensor

                        base_name = self.name_map.get(key, key)
                        path_root = os.path.join(item_dir, base_name)
                        self._save_item(sample_item, path_root)
