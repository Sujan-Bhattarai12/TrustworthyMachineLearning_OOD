
import csv
import pickle
from pathlib import Path

import numpy as np
import torch


DATA_DIR = Path("data/cifar-10-batches-py")
MANIFEST_PATH = Path("splits/split_manifest_v2.csv")


class CIFARDataLoader:
    def __init__(self, manifest_path=MANIFEST_PATH):
        # Load all 50,000 training images in their original order
        train_batches = []

        for batch_id in range(1, 6):
            with open(
                DATA_DIR / f"data_batch_{batch_id}", "rb"
            ) as f:
                batch = pickle.load(f, encoding="bytes")

            train_batches.append(batch[b"data"])

        self.train_images = np.concatenate(
            train_batches, axis=0
        )

        # Load official 10,000 test images
        with open(DATA_DIR / "test_batch", "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")

        self.test_images = test_batch[b"data"]

        # Load active experiment manifest
        with open(manifest_path, newline="") as f:
            self.manifest = list(csv.DictReader(f))

        assert self.train_images.shape == (50000, 3072)
        assert self.test_images.shape == (10000, 3072)

    def get_image(self, image_id, source, device="cpu"):
        """
        Return one CIFAR-10 image as a float tensor.

        Shape: (1, 3, 32, 32)
        Values: [0, 1]
        """

        if source == "train":
            raw = self.train_images[image_id]
        elif source == "test":
            raw = self.test_images[image_id]
        else:
            raise ValueError(
                "source must be 'train' or 'test'"
            )

        image = raw.reshape(3, 32, 32)

        return torch.tensor(
            image.astype(np.float32) / 255.0,
            device=device,
        ).unsqueeze(0)

    def get_split(self, split):
        """Return manifest rows for a given split."""
        return [
            row for row in self.manifest
            if row["split"] == split
        ]
