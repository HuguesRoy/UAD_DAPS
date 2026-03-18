from clinicadl.data.datasets import CapsDataset
from clinicadl.data.dataloader import DataLoaderConfig

def CapsDataset_with_conversion(*args, conversion_arg=None, **kwargs):
    dataset = CapsDataset(*args, **kwargs)
    dataset.read_tensor_conversion(conversion_arg)
    return dataset

class DataLoaderWrapper:
    """
    Hydra-friendly wrapper for ClinicaDL DataLoader.
    Makes it behave like torch DataLoader partials when used with Hydra.
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

    def __call__(self, dataset):
        
        dataloader_cfg = DataLoaderConfig(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
        )
        return dataloader_cfg.get_object(dataset)
