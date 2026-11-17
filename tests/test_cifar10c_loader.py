import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cifar10c_loader import CIFAR10CLoader

class CIFAR10CLoadingTests(unittest.TestCase):
    def fixture(self,folder,source='test'):
        manifest=Path(folder)/'manifest.csv'
        manifest.write_text(f'image_id,class_id,source,split\n23,3,{source},development\n')
        clean=Path(folder)/'test_batch';clean.write_bytes(b'fixture')
        labels=np.arange(10000,dtype=np.uint8)%10
        batch={b'data':np.broadcast_to(np.zeros((1,3072),dtype=np.uint8),(10000,3072)),b'labels':labels}
        # Virtual 50k-image array: unique constant pixel values identify each row.
        pixels=np.broadcast_to((np.arange(50000)%256).astype(np.uint8)[:,None,None,None],(50000,32,32,3))
        def load(path,**kwargs):return np.tile(labels,5) if Path(path).name=='labels.npy' else pixels
        return manifest,clean,batch,load

    def test_official_severity_block_index_and_clean_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,clean,batch,load=self.fixture(folder)
            with patch('src.cifar10c_loader.pickle.load',return_value=batch),patch('src.cifar10c_loader.np.load',side_effect=load):
                loader=CIFAR10CLoader(manifest,folder,clean)
                for level in range(1,6):
                    x=loader.get_corrupted_image(23,'test','brightness',level)
                    self.assertEqual(tuple(x.shape),(1,3,32,32))
                    self.assertAlmostEqual(float(x.mean()),(((level-1)*10000+23)%256)/255,places=6)
                self.assertEqual(float(loader.get_corrupted_image(23,'test','brightness',0).sum()),0.)

    def test_rejects_training_source_and_bad_severity(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,clean,batch,load=self.fixture(folder,'train')
            with patch('src.cifar10c_loader.pickle.load',return_value=batch),patch('src.cifar10c_loader.np.load',side_effect=load):
                with self.assertRaises(ValueError):CIFAR10CLoader(manifest,folder,clean)
            manifest,clean,batch,load=self.fixture(folder)
            with patch('src.cifar10c_loader.pickle.load',return_value=batch),patch('src.cifar10c_loader.np.load',side_effect=load):
                loader=CIFAR10CLoader(manifest,folder,clean)
                with self.assertRaises(ValueError):loader.get_corrupted_image(23,'test','brightness',6)
                with self.assertRaises(ValueError):loader.get_corrupted_image(23,'test','blur',1)

    def test_label_alignment_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,clean,batch,load=self.fixture(folder)
            def bad(path,**kwargs):return np.zeros(50000,dtype=np.uint8)
            with patch('src.cifar10c_loader.pickle.load',return_value=batch),patch('src.cifar10c_loader.np.load',side_effect=bad):
                with self.assertRaises(ValueError):CIFAR10CLoader(manifest,folder,clean)
