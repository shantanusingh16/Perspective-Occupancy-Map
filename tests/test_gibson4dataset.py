import os
import torch
import numpy as np
import cv2
import unittest
from pyfakefs.fake_filesystem_unittest import TestCase
from tests import DataConfig

from datasets.gibson4 import Gibson4Dataset

class TestGibson4Dataset(TestCase):

    def setUp(self) -> None:
        self.setUpPyfakefs()
        self.dataconfig = DataConfig(self.fs)

        self.train_dataset = Gibson4Dataset(self.dataconfig.data_dir, self.dataconfig.train_files, True)
        self.test_dataset = Gibson4Dataset(self.dataconfig.data_dir, self.dataconfig.test_files, False)
           
    def test_checksizes(self):
        self.assertEqual(len(self.train_dataset), len(self.dataconfig.train_files))
        self.assertEqual(len(self.test_dataset), len(self.dataconfig.test_files))

    def test_output_shape(self):
        for item in self.train_dataset:
            self.assertEqual(len(item), 4)
            rgb, sem, pom, bev = item
            self.assertEqual(tuple(rgb.shape), (3, *self.train_dataset.img_size))
            self.assertEqual(tuple(sem.shape), (1, *self.train_dataset.img_size))
            self.assertEqual(tuple(pom.shape), (1, *self.train_dataset.tgt_size))
            self.assertEqual(tuple(bev.shape), (1, *self.dataconfig.fname_metadata_map['partial_occ'][1]))
        
        for item in self.test_dataset:
            self.assertEqual(len(item), 4)
            rgb, sem, pom, bev = item
            self.assertEqual(tuple(rgb.shape), (3, *self.test_dataset.img_size))
            self.assertEqual(tuple(sem.shape), (1, *self.test_dataset.img_size))
            self.assertEqual(tuple(pom.shape), (1, *self.test_dataset.tgt_size))
            self.assertEqual(tuple(bev.shape), (1, *self.dataconfig.fname_metadata_map['partial_occ'][1]))

    def test_colordiff_train(self):
        rgb0, sem0, pom0, bev0 = self.train_dataset[0]

        rgb1, sem1, pom1, bev1 = self.train_dataset[0]
        self.assertEqual(torch.isclose(rgb0, rgb1).all(), False)  # Due to color jitter
        self.assertEqual(torch.isclose(sem0, sem1).all(), True)
        self.assertEqual(torch.isclose(pom0, pom1).all(), True)
        self.assertEqual(torch.isclose(bev0, bev1).all(), True)


        rgb1, sem1, pom1, bev1 = self.train_dataset[1]
        self.assertEqual(torch.isclose(rgb0, rgb1).all(), False)
        self.assertEqual(torch.isclose(sem0, sem1).all(), False)
        self.assertEqual(torch.isclose(pom0, pom1).all(), False)
        self.assertEqual(torch.isclose(bev0, bev1).all(), False)

    def test_colordiff_test(self):
        rgb0, sem0, pom0, bev0 = self.test_dataset[0]

        rgb1, sem1, pom1, bev1 = self.test_dataset[0]
        self.assertEqual(torch.isclose(rgb0, rgb1).all(), True)  # No color jitter for test
        self.assertEqual(torch.isclose(sem0, sem1).all(), True)
        self.assertEqual(torch.isclose(pom0, pom1).all(), True)
        self.assertEqual(torch.isclose(bev0, bev1).all(), True)

        rgb1, sem1, pom1, bev1 = self.test_dataset[1]
        self.assertEqual(torch.isclose(rgb0, rgb1).all(), False)
        self.assertEqual(torch.isclose(sem0, sem1).all(), False)
        self.assertEqual(torch.isclose(pom0, pom1).all(), False)
        self.assertEqual(torch.isclose(bev0, bev1).all(), False)

if __name__ == '__main__':
    unittest.main()
