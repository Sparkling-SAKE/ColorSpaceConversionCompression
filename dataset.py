# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from neuralcompression.data import Vimeo90kSeptuplet

class Kodak24Dataset(Dataset):
    def __init__(self, root, transform=None, split="kodak24"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class OpenImageDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # Openimage
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")

        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    
class OpenImageLightning(LightningDataModule):
    """
    PyTorch Lightning data module version of ``Vimeo90kSeptuplet``.

    Args:
        data_dir: root directory of Vimeo dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This improves performance on GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256,256),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(self.patch_size), transforms.ToTensor()]
        )

        val_transforms = transforms.Compose(
            [transforms.CenterCrop(self.patch_size), transforms.ToTensor()]
        )

        self.train_dataset = OpenImageDataset(
            self.data_dir,
            transform=train_transforms,
            split="train",
        )

        self.val_dataset = OpenImageDataset(
            self.data_dir,
            transform=val_transforms,
            split="val",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
        
class Vimeo90kSeptupletLightning(LightningDataModule):
    """
    PyTorch Lightning data module version of ``Vimeo90kSeptuplet``.

    Args:
        data_dir: root directory of Vimeo dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This improves performance on GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(self.patch_size), transforms.ToTensor()]
        )

        val_transforms = transforms.Compose(
            [transforms.CenterCrop(self.patch_size), transforms.ToTensor()]
        )

        self.train_dataset = Vimeo90kSeptuplet(
            self.data_dir,
            pil_transform=train_transforms,
            split="train",
        )

        self.val_dataset = Vimeo90kSeptuplet(
            self.data_dir,
            pil_transform=val_transforms,
            split="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
