import glob
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import augmentations


class ImdbDataset(Dataset):
    def __init__(self, source_root, split=(0.0, 0.9)):
        image_paths = sorted(glob.glob(os.path.join(source_root, '**/*.jpg')))
        self.image_paths = image_paths[int(len(image_paths) * split[0]):int(len(image_paths) * split[1])]
        with open(os.path.join(source_root, 'landmarks.pkl'), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.size = 256

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def resize(cls, image, size):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    @classmethod
    def transform(cls, img, size=256, norm=True):
        if img.shape[0] != size and img.shape[1] != size:
            img = cls.resize(img, size)
        img = img.astype(np.float32) / 255.
        if norm:
            img = (img - 0.5) / 0.5
        return img

    def denorm_lmarks(self, landmarks, img):
        landmarks = landmarks.copy()
        landmarks[:, 0] = landmarks[:, 0] * img.shape[1]
        landmarks[:, 1] = landmarks[:, 1] * img.shape[0]
        return landmarks

    def __getitem__(self, index):
        from_path = self.image_paths[index]
        from_im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGB)
        src_path = '/'.join(from_path.split('/')[-2:])
        landmark = self.denorm_lmarks(self.landmarks[src_path], from_im)
        msk = landmark[1:16,:2].copy()
        msk = msk.astype(np.uint32)
        masked = cv2.fillPoly(from_im,msk,(0,0,0))
        from_im = self.transform(from_im, self.size)
        masked = self.transform(masked, self.size)
        from_im = torch.tensor(from_im).permute([2, 0, 1])
        masked = torch.tensor(masked).permute([2, 0, 1])
        return masked, from_im