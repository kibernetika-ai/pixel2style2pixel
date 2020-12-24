import glob
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImdbDataset(Dataset):
    def __init__(self, source_root, source_transform=None):
        self.image_paths = glob.glob(os.path.join(source_root, '**/*.jpg'))
        with open(os.path.join(source_root, 'landmarks.pkl'), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.source_transform = source_transform
        self.size = 256

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image):
        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)  # flip horizontally

        img = img.astype(np.float32) / 255.
        img = (img - 0.5) / 0.5
        return img

    def __getitem__(self, index):
        from_path = self.image_paths[index]
        from_im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGB)

        to_path = np.random.choice(self.image_paths)
        while to_path == from_path:
            to_path = np.random.choice(self.image_paths)

        to_im = cv2.cvtColor(cv2.imread(to_path), cv2.COLOR_BGR2RGB)
        from_im = self.transform(from_im)
        to_im = self.transform(to_im)

        return torch.tensor(from_im), torch.tensor(to_im)
