import torch
from typing import Dict, Optional
import torch.nn as nn

class BaseWrapper:
    def __call__(self, batch: Dict) -> Dict:
        raise NotImplementedError


class BratsDataWrapper(BaseWrapper):
    def __init__(self,
        device: str,
        image_key: str = "image",
        mask_key: Optional[str] = None,
        seg_key: Optional[str] = None,
        label_key: Optional[str] = None,
        seg_threshold = 0,
        mode_threshold = "greater"
    ):

        self.device = device
        self.image_key = image_key
        self.mask_key = mask_key
        self.seg_key = seg_key
        self.label_key = label_key

        self.seg_threshold = seg_threshold
        self.mode_threshold = mode_threshold

    def __call__(self, batch: Dict) -> Dict:
        output = {}

        # Required image

        try:
            output["image"] = batch[self.image_key].to(self.device)
        except KeyError:
            raise KeyError(
                f"Expected image key '{self.image_key}' in batch, but not found."
            )

        # Optional mask
        output["mask"] = (
            batch[self.mask_key].to(self.device)
            if self.mask_key and self.mask_key in batch
            else None
        )

        # Optional segmentation

        if self.seg_key and self.seg_key in batch:
            seg = batch[self.seg_key]

            if self.mode_threshold == "greater":
                seg_mask = (seg > self.seg_threshold).to(torch.int8)
            else:
                seg_mask = (seg < self.seg_threshold).to(torch.int8)

            output["seg"] = seg.to(self.device)          # keep original segmentation
            output["seg_mask"] = seg_mask.to(self.device)  # threshold mask
        else:
            output["seg"] = None
            output["seg_mask"] = None

        # Optional label
        output["label"] = (
            batch[self.label_key].to(self.device)
            if self.label_key and self.label_key in batch
            else None
        )

        return output

class ClinicaDLWrapper:
    """
    Configurable wrapper for ClinicaDL dataset batches.
    Maps dataset keys (user-defined) to a normalized dict with:
        - x: image tensor
        - mask: brainmask tensor (optional, else None)
        - seg: segmentation tensor (optional, else None)
        - label: label tensor (optional, else None)
        - participant_id, session_id: metadata (kept as-is)

    Example:
        wrapper = ClinicaDLWrapper(
            device="cuda",
            image_key="t1_linear",
            mask_key="brainmask",
            seg_key="segmentation",
            label_key="label",
        )
    """

    def __init__(
        self,
        device: str,
        image_key: str = "image",
        mask_key: Optional[str] = None,
        seg_key: Optional[str] = None,
        label_key: Optional[str] = None,
        seg_threshold = 0,
        mode_threshold = "greater",
        original_image_key: Optional[str] = None
    ):
        self.device = device
        self.image_key = image_key
        self.mask_key = mask_key
        self.seg_key = seg_key
        self.label_key = label_key

        self.seg_threshold = seg_threshold
        self.mode_threshold = mode_threshold
        self.original_image_key = original_image_key

    def __call__(self, batch: Dict) -> Dict:
        output = {}

        # Required image

        try:
            output["image"] = batch.get_field(self.image_key).to(self.device)
        except KeyError:
            raise KeyError(
                f"Expected image key '{self.image_key}' in batch, but not found."
            )

        # Optional mask
        output["mask"] = (
            batch.get_field(self.mask_key).to(self.device)
            if self.mask_key
            else None
        )

        # Optional segmentation
        if self.seg_key:
            seg = batch.get_field(self.seg_key)

            if self.mode_threshold == "greater":
                seg_mask = (seg > self.seg_threshold).to(torch.int8)
            else:
                seg_mask = (seg < self.seg_threshold).to(torch.int8)

            output["seg"] = seg.to(self.device)          # keep original segmentation
            output["seg_mask"] = seg_mask.to(self.device)  # threshold mask
        else:
            output["seg"] = None
            output["seg_mask"] = None

        # Optional label
        output["label"] = (
            batch.get_field(self.label_key).to(self.device)
            if self.label_key
            else None
        )

        # Optional original image
        if self.original_image_key:
            try:
                output["true_healthy"] = batch.get_field(self.original_image_key).to(self.device)
            except KeyError:
                pass
        # Metadata (not moved to device)
        for meta_key in ["participant_id", "session_id"]:
            if meta_key in batch:
                output[meta_key] = batch[meta_key]

        return output


class TransformVQVAEWrapper(BaseWrapper):
    def __init__(self, base_wrapper: BaseWrapper, transform : nn.Module):
        self.base_wrapper = base_wrapper
        self.transform = transform

    def __call__(self, batch):
        data = self.base_wrapper(batch)
        with torch.no_grad():
            embeddings = self.transform.encoder(data["image"])
            data["image"] = self.transform.quantizer(embeddings)["quantized_st"]
        return data


class TransformWrapper(BaseWrapper):
    def __init__(self, base_wrapper: BaseWrapper, transform: nn.Module):
        self.base_wrapper = base_wrapper
        self.transform = transform

    def __call__(self, batch):
        data = self.base_wrapper(batch)
        with torch.no_grad():
            data["image"] = self.transform(data["image"])
        return data


class TransformMultiVae(BaseWrapper):
    def __init__(self, base_wrapper: BaseWrapper, transform: nn.Module):
        self.base_wrapper = base_wrapper
        self.transform = transform
        self.transform = self.transform.encoder
        self.transform.to(base_wrapper.device)
        self.transform.eval()


    def __call__(self, batch):
        data = self.base_wrapper(batch)
        with torch.no_grad():
            embeddings = (self.transform(data["image"])).embedding
            data["image"] = embeddings.unsqueeze(1)
        return data

class TransformMultiVaeVQVAE(BaseWrapper):
    def __init__(self, base_wrapper: BaseWrapper, transform: nn.Module, use_quantizer: bool = True):
        self.base_wrapper = base_wrapper
        self.transform = transform
        self.transform.to(base_wrapper.device)
        self.transform.eval()

        self.use_quantizer = use_quantizer

    def __call__(self, batch):
        data = self.base_wrapper(batch)
        with torch.no_grad():

            embeddings = (self.transform.encoder(data["image"])).embedding

            if self.use_quantizer:
                embeddings, reshape_for_decoding = self.transform._reshape_for_quantizer(
                    embeddings, self.transform.model_config
                )

                quantizer_output = self.transform.quantizer(embeddings, uses_ddp=False)

                quantized_embed = quantizer_output.quantized_vector

                data["image"] = quantized_embed
            else:
                data["image"] = embeddings
        return data