from typing import Tuple
import torch
import torch.nn.functional as F
import torchio as tio
from clinicadl.data.structures import DataPoint
from omegaconf import ListConfig
from copy import deepcopy
from pathlib import Path
from typing import Union
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


class SliceResize(tio.IntensityTransform):
    """
    Resize 2D slices of a DataPoint to a given spatial size.

    Automatically uses 'nearest' for LabelMap and the configured
    interpolation mode (e.g. 'bilinear') for ScalarImage.

    Parameters
    ----------
    size : Tuple[int, int]
        Target (height, width) for each slice.
    mode : str, optional (default="bilinear")
        Interpolation mode for ScalarImage.
    align_corners : bool, optional (default=False)
        Passed to torch.nn.functional.interpolate for "linear" modes.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mode: str = "bilinear",
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = tuple(size)
        self.mode = mode
        self.align_corners = align_corners
        self.args_names = ["size", "mode", "align_corners"]

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        # Apply to all images, not just intensity
        for image in datapoint.get_images(intensity_only=False):
            tensor = image.tensor.squeeze(datapoint.slice_direction + 1)
            resized = self._resize(tensor, datapoint.slice_direction, image)
            image.set_data(resized)
        return datapoint

    def _resize(
        self, tensor: torch.Tensor, slice_direction: int, image: tio.Image
    ) -> torch.Tensor:
        """Resize tensor with interpolation mode depending on image type."""
        # Pick interpolation mode
        if isinstance(image, tio.LabelMap):
            interp_mode = "nearest"
            align_corners = None
        else:
            interp_mode = self.mode
            align_corners = self.align_corners if "linear" in self.mode else None

        tensor = tensor.float()

        # Handle 3D (C, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            out = F.interpolate(
                tensor, size=self.size, mode=interp_mode, align_corners=align_corners
            )
            return out.squeeze(0).unsqueeze(slice_direction + 1)

        # Handle 4D (C, D, H, W)
        elif tensor.ndim == 4:
            B, D = tensor.shape[0], tensor.shape[1]
            tensor = tensor.reshape(B * D, 1, tensor.shape[2], tensor.shape[3])
            out = F.interpolate(
                tensor, size=self.size, mode=interp_mode, align_corners=align_corners
            )
            out = out.reshape(B, D, *self.size)
            return out.unsqueeze(slice_direction + 1)

        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")




class NormalizationBrats(object):
    """Normalizes a tensor between min and max"""

    def __init__(self, min, max):
       self.min = min
       self.max = max
   

    def __call__(self, image):
        return (self.max - self.min)*((image - image.min()) / (image.max() - image.min())) + self.min
    

class ResizeInterpolationBrats(object):
  """ Interpolation  """
   
  def __init__(self, size: tuple, mode : str = 'trilinear', align_corners : bool = False):
      self.size = tuple(size)
      self.mode = mode
      self.align_corners = align_corners

  def __call__(self,image : torch.Tensor) -> torch.Tensor:
    
    image  = image.unsqueeze(0)
    image = F.interpolate(image, size= self.size, mode= self.mode, align_corners= self.align_corners).squeeze(0)
    return image


class Hypometabolism(tio.IntensityTransform):
    """
    To simulate dementia-related hypometabolism, aimed to be used on FDG PET images.
    This transform reduces the intensity of signal in specific brain regions based on the mask of a pathology.
    Ref. https://www.melba-journal.org/papers/2024:003.html.
    Parameters
    ----------
    mask_dir : str or Path
        Directory containing the mask files.
    pathology : str
        Type of pathology to simulate (e.g., "AD", "bvFTD").
    percentage : int
        Percentage of hypometabolism to apply.
        The value should be between 0 and 100.
        0 means no hypometabolism, while 100 means complete removal of the signal in the masked regions.
    sigma : int, optional
        Standard deviation for Gaussian noise, by default 2.
    """

    def __init__(
        self,
        mask_dir: Union[str, Path],
        pathology: str,
        percentage: int,
        sigma: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")

        self.mask_dir = mask_dir
        self.pathology = pathology
        self.percentage = percentage
        self.sigma = sigma

        mask_path = Path(
            self.mask_dir, f"mask_hypo_{self.pathology.lower()}_resampled.nii"
        )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Mask file for pathology '{self.pathology}' not found at {mask_path}."
            )
        mask_nii = nib.load(mask_path)
        self.binary_mask = mask_nii.get_fdata()
        self.mask = self._mask_processing(self.binary_mask)

        self.args_names = ["mask_dir", "pathology", "percentage", "sigma"]

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:  # pylint: disable=arguments-renamed
        """
        Apply the transform to the datapoint.
        """
        transformed = deepcopy(datapoint)

        for image in transformed.get_images(intensity_only=True):
            image.tensor[:] *= self.mask

        transformed.add_image(datapoint.image, "original_image")
        transformed.add_mask(
            tio.LabelMap(
                tensor=np.expand_dims(self.mask, axis=0), affine=datapoint.image.affine
            ),
            "hypo_mask",
        )
        transformed.add_mask(
            tio.LabelMap(
                tensor=np.expand_dims(self.binary_mask, axis=0),
                affine=datapoint.image.affine,
            ),
            "binary_hypo_mask",
        )
                
        transformed["pathology"] = self.pathology
        transformed["percentage"] = self.percentage

        return transformed

    def _mask_processing(self, mask: np.array) -> np.array:
        inverse_mask = 1 - mask
        inverse_mask[inverse_mask == 0] = 1 - self.percentage / 100
        gaussian_mask = gaussian_filter(inverse_mask, sigma=self.sigma)
        return np.float32(gaussian_mask)