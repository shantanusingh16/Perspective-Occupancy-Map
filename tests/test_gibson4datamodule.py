import unittest
from pyfakefs.fake_filesystem_unittest import TestCase
from datasets.gibson4 import Gibson4DataModule
from tests import DataConfig


class TestGibson4Dataset(TestCase):

    def setUp(self) -> None:
        self.setUpPyfakefs()
        self.dataconfig = DataConfig(self.fs)
        self.num_batches = 2
        self.batch_size = len(self.dataconfig.train_files) // self.num_batches
        self.train_val_split = 0.8
        self.dm = Gibson4DataModule(self.dataconfig.data_dir, self.dataconfig.split_dir, \
            self.train_val_split, self.batch_size)
        self.dm.setup()
           
    def test_checksizes(self):

        train_len = int(len(self.dataconfig.train_files) * self.train_val_split)
        val_len = len(self.dataconfig.train_files) - train_len

        self.assertEqual(len(self.dm.gibson4_train), train_len)
        self.assertEqual(len(self.dm.gibson4_val), val_len)

        self.assertEqual(len(self.dm.gibson4_test), len(self.dataconfig.test_files))

    def test_numbatches(self):
        self.assertEqual(len(self.dm.train_dataloader()), self.num_batches)

    def test_batchsize(self):
        train_batch = next(self.dm.train_dataloader()._get_iterator())
        self.assertEqual(len(train_batch), 4)  # RGB, SEM, POM, BEV
        for idx in range(len(train_batch)):
            self.assertEqual(train_batch[idx].shape[0], self.batch_size)

        test_batch = next(self.dm.test_dataloader()._get_iterator())
        self.assertEqual(len(test_batch), 4)  # RGB, SEM, POM, BEV
        for idx in range(len(test_batch)):
            self.assertLessEqual(test_batch[idx].shape[0], self.batch_size)

    

if __name__ == '__main__':
    unittest.main()
