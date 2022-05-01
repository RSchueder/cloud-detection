import numpy as np
import xarray as xr
from PIL import Image
from tensorflow import keras

from cloud_detection.config import NUM_CLASSES, NUM_BANDS


class DataGenerator(keras.utils.Sequence):
    """
    Class to create samples and target data.
    """

    def __init__(self, batch_size, img_size, sample_paths, target_paths):
        """

        Args:
            batch_size (int): The batch size.
            img_size (tuple(int, int)): The input image size.
            sample_paths (list): A list of paths to sample data.
            target_paths (list): A list of paths to target data.
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.sample_paths = sample_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.target_paths) // self.batch_size

    def __getitem__(self, idx):

        pos = idx * self.batch_size
        batch_sample_paths = self.sample_paths[pos:pos + self.batch_size]
        batch_target_paths = self.target_paths[pos:pos + self.batch_size]

        X = np.zeros((self.batch_size,) + self.img_size + (NUM_BANDS,), dtype="float32")
        for idx, path in enumerate(batch_sample_paths):
            dataset = xr.open_dataset(path)
            img = dataset.band_data.values
            img = np.moveaxis(img, [0], [2])
            X[idx] = img

        target = np.zeros((self.batch_size,) + self.img_size + (NUM_CLASSES,), dtype="uint8")
        for idx, path in enumerate(batch_target_paths):

            image = Image.open(path)
            image = np.asarray(image)
            one_hot_mask_data = np.zeros([NUM_CLASSES] + list(image.shape))
            for pclass in range(NUM_CLASSES):
                pslice = one_hot_mask_data[pclass, :, :]
                pslice[image == pclass] = 1
                one_hot_mask_data[pclass, :, :] = pslice

            target[idx] = np.moveaxis(one_hot_mask_data, [0], [2])

        return X, target
