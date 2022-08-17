import os
from typing import List
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

g = torch.Generator()
g.manual_seed(42)


class Gibson4Dataset(Dataset):
    def __init__(self, data_dir:str, files:List[str], train: bool=False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.img_size = (512, 512)
        self.tgt_size = (128, 128)
        
        # TODO: Create a mapping from semantic indices to objectnav and supplementary target indices.
        # Objectnav categories - chair, couch, potted plant, bed, toilet and tv
        # Supplementary categories - cabinet, floor, table, shelf, desk, carpet, wardrobe, lamp, refrigerator, stairs, door, dishwasher, microwave
        # self.object_idx_map = {
        #     1: 1,  # Wall
        #     4: 2,  # Floor
        #     8: 3,  # Bed
        #     11: 4, # Cabinet
        # }

        color_transforms = [
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]
        if not train:
            color_transforms.pop(0)

        self.color_transforms = transforms.Compose(color_transforms)
        self.semantics_transforms = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.lbl_transforms = transforms.Compose([
            transforms.Resize(self.tgt_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        scene, camera, fileidx = self.files[index].split()

        rgb_path = os.path.join(self.data_dir, scene, '0', camera, 'RGB', f'{fileidx}.jpg')
        rgb = self.color_transforms(Image.open(rgb_path))

        sem_path = os.path.join(self.data_dir, scene, '0', camera, 'semantics', f'{fileidx}.png')
        sem = self.semantics_transforms(Image.open(sem_path))

        sem_pom_path = os.path.join(self.data_dir, scene, '0', camera, 'pom', f'{fileidx}.png')
        sem_pom = np.array(Image.open(sem_pom_path))

        # Convert semantic perspective occupancy map to simple occupancy map
        pom = np.zeros_like(sem_pom)
        pom[sem_pom != 0] = 1
        pom[np.isin(sem_pom, [4, 29])] = 2
        pom = self.lbl_transforms(Image.fromarray(pom))   

        bev_path = os.path.join(self.data_dir, scene, '0', camera, 'partial_occ', f'{fileidx}.png')
        bev = torch.from_numpy(np.array(Image.open(bev_path), dtype=int) // 127)

        return rgb, sem, pom, bev



class Gibson4DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, split_dir: str, train_val_split: float = 0.9, batch_size: int = 32, num_workers: int=0):
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage=None):
        with open(os.path.join(self.split_dir, 'train.txt'), 'r') as f:
            train_files = f.read().splitlines()
        with open(os.path.join(self.split_dir, 'test.txt'), 'r') as f:
            test_files = f.read().splitlines()

        if stage == 'fit' or stage is None:
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
            shuffle=True, generator=g, drop_last=True, \
            num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.gibson4_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.gibson4_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.gibson4_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def teardown(self, stage=None):
        super(Gibson4DataModule, self).teardown(stage)