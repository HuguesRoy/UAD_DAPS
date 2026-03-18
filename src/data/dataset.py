import torch
from torch.utils.data import Dataset
from pathlib import Path
import os 
import numpy as np

class BraTSSanityDataset(Dataset):
    def __init__(self, data_root: str, mode: str, input_modality, transforms=None):
        super(BraTSSanityDataset, self).__init__()

        assert mode in ['train', 'val','test'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_modality = input_modality

        self.transforms = transforms
        self.case_names_input = sorted(list(Path(os.path.join(self.data_root, input_modality)).iterdir()))
        self.case_names_brainmask = sorted(list(Path(os.path.join(self.data_root, 'brainmask')).iterdir()))
        self.case_names_seg = sorted(list(Path(os.path.join(self.data_root, 'seg')).iterdir()))

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index].name
        name_brainmask = self.case_names_brainmask[index].name
        name_seg = self.case_names_seg[index].name
        base_dir_input = os.path.join(self.data_root, self.input_modality, name_input)
        base_dir_brainmask = os.path.join(self.data_root, 'brainmask', name_brainmask)
        base_dir_seg = os.path.join(self.data_root, 'seg', name_seg)

        input = torch.from_numpy(np.load(base_dir_input).astype(np.float32)).unsqueeze(0)
        brain_mask =  torch.from_numpy(np.load(base_dir_brainmask).astype(np.float32)).unsqueeze(0)
        seg =  torch.from_numpy(np.load(base_dir_seg).astype(np.float32)).unsqueeze(0)
        
        seg = self.transforms(seg)
        seg = (seg > seg.min())

        item = {'image': self.transforms(input), 'brainmask': self.transforms(brain_mask), 'seg': seg}

        return item

    def __len__(self):
        return len(self.case_names_input)