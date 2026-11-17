"""Load pre-generated CIFAR-10-C pixels, paired with original clean test images.

Official layout: (severity-1)*10000 + original CIFAR-10 test image ID.
Never regenerates or interpolates benchmark corruptions.
"""
import csv
import pickle
from pathlib import Path
import numpy as np
import torch

CIFAR10C_CORRUPTIONS=('brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_blur',
    'gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','saturate',
    'shot_noise','snow','spatter','speckle_noise','zoom_blur')

class CIFAR10CLoader:
    def __init__(self,manifest_path,corruption_dir='data/CIFAR-10-C',clean_test_path='data/cifar-10-batches-py/test_batch'):
        self.corruption_dir=Path(corruption_dir)
        with Path(clean_test_path).open('rb') as stream:batch=pickle.load(stream,encoding='bytes')
        self.test_images=np.asarray(batch[b'data']);self.test_labels=np.asarray(batch[b'labels'])
        if self.test_images.shape!=(10000,3072) or self.test_labels.shape!=(10000,):raise ValueError('Unexpected clean CIFAR-10 test layout.')
        labels=np.load(self.corruption_dir/'labels.npy',mmap_mode='r',allow_pickle=False)
        if labels.shape!=(50000,) or not all(np.array_equal(labels[i*10000:(i+1)*10000],self.test_labels) for i in range(5)):
            raise ValueError('CIFAR-10-C labels do not match clean test ordering.')
        with Path(manifest_path).open(newline='') as stream:self.manifest=list(csv.DictReader(stream))
        if any(r.get('source')!='test' for r in self.manifest):raise ValueError('CIFAR-10-C pairs only with original test-source images.')
        for row in self.manifest:
            image=int(row['image_id'])
            if not 0<=image<10000:raise ValueError('Invalid test image ID.')
            label=int(row.get('class_id',row.get('true_label',-1)))
            if label!=int(self.test_labels[image]):raise ValueError('Manifest label mismatch.')
        self.arrays={}

    def get_split(self,split):return [row for row in self.manifest if row['split']==split]

    def get_image(self,image_id,source,device='cpu'):
        if source!='test' or not 0<=int(image_id)<10000:raise ValueError('Expected test-source ID 0–9999.')
        image=self.test_images[int(image_id)].reshape(3,32,32)
        return torch.as_tensor(image.astype(np.float32)/255,device=device).unsqueeze(0)

    def get_corrupted_image(self,image_id,source,corruption,severity,device='cpu'):
        if source!='test' or not 0<=int(image_id)<10000:raise ValueError('Expected test-source ID 0–9999.')
        if corruption not in CIFAR10C_CORRUPTIONS:raise ValueError('Unknown official corruption.')
        if severity not in range(6):raise ValueError('Severity must be 0–5.')
        if severity==0:return self.get_image(image_id,source,device)
        if corruption not in self.arrays:
            array=np.load(self.corruption_dir/f'{corruption}.npy',mmap_mode='r',allow_pickle=False)
            if array.shape!=(50000,32,32,3) or array.dtype!=np.uint8:raise ValueError('Unexpected official corruption file layout.')
            self.arrays[corruption]=array
        array=self.arrays[corruption][(severity-1)*10000+int(image_id)]
        return torch.as_tensor(array.transpose(2,0,1).astype(np.float32)/255,device=device).unsqueeze(0)
