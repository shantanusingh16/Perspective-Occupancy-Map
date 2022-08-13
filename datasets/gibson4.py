from configparser import Interpolation
import os
from posixpath import split
from typing import List
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from PIL import Image
import torchvision.transforms as transforms

g = torch.Generator()
g.manual_seed(42)


class Gibson4Dataset(Dataset):
    def __init__(self, data_dir:str, files:List[str], train: bool=False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.img_size = (128, 128)

        color_transforms = [
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        if not train:
            color_transforms.pop(0)

        self.color_transforms = transforms.Compose(color_transforms)

        self.lbl_transforms = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        scene, camera, fileidx = self.files[index].split()

        rgb_path = os.path.join(self.data_dir, scene, '0', camera, 'RGB', f'{fileidx}.jpg')
        rgb = self.color_transforms(Image.open(rgb_path))

        sem_path = os.path.join(self.data_dir, scene, '0', camera, 'semantics', f'{fileidx}.png')
        sem = self.lbl_transforms(Image.open(sem_path))

        pom_path = os.path.join(self.data_dir, scene, '0', camera, 'pom', f'{fileidx}.png')
        pom = self.lbl_transforms(Image.open(pom_path))

        bev_path = os.path.join(self.data_dir, scene, '0', camera, 'partial_occ', f'{fileidx}.png')
        bev = transforms.ToTensor()(Image.open(bev_path))

        return rgb, sem, pom, bev



class Gibson4DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, split_dir: str, train_val_split: float = 0.9, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage=None):
        with open(os.path.join(self.split_dir, 'train.txt'), 'r') as f:
            train_files = f.read().splitlines()
        with open(os.path.join(self.split_dir, 'test.txt'), 'r') as f:
            test_files = f.read().splitlines()

        if stage == 'train' or stage == 'val' or stage is None:
            dataset = Gibson4Dataset(self.data_dir, train_files, train=True)
            train_size = int(len(dataset) * self.train_val_split)
            val_size = len(dataset) - train_size

            gibson4_train, gibson4_val = random_split(dataset, [train_size, val_size])
            self.gibson4_train = gibson4_train
            self.gibson4_val = gibson4_val

        if stage == 'test' or stage == 'predict' or stage is None:
            self.gibson4_test = Gibson4Dataset(self.data_dir, test_files)

    def train_dataloader(self):
        return DataLoader(self.gibson4_train, batch_size=self.batch_size, \
            shuffle=True, generator=g, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.gibson4_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.gibson4_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.gibson4_test, batch_size=self.batch_size)

    def teardown(self, stage=None):
        super(Gibson4DataModule, self).teardown(stage)