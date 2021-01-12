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
        #self.orig_src = '/notebooks/imdb'
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
        landmarks = np.clip(landmarks,0,256)
        return landmarks

    def forehead_coords(self, p):
        x0 = (p[0][0] + p[17][0]) / 2
        x1 = (p[16][0] + p[26][0]) / 2
        y0 = np.max(p[17:27, 1])
        height = np.max(p[27:36, 1]) - np.min(p[0:17, 1])
        return np.array([x0, max(y0 - height, 0), x1, y0])

    def __getitem__(self, index):
        from_path = self.image_paths[index]
        from_im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGB)
        src_path = '/'.join(from_path.split('/')[-2:])
        #orig_img = os.path.join(self.orig_src,src_path)
        #orig_img = cv2.imread(orig_img)
        landmark = self.denorm_lmarks(self.landmarks[src_path], from_im)
        msk = landmark[0:17,:2].copy()
        msk = msk.astype(np.int32)
        msk = np.array([msk],np.int32)
        masked = cv2.fillPoly(from_im.copy(),msk,(0,0,0))
        from_im = self.transform(from_im, self.size)
        masked = self.transform(masked, self.size)
        from_im = torch.from_numpy(from_im).permute([2, 0, 1])
        masked = torch.from_numpy(masked).permute([2, 0, 1])
        fc = self.forehead_coords(landmark).astype(np.int32)
        return masked, from_im, torch.from_numpy(fc)