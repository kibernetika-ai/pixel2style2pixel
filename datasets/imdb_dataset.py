import glob
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import augmentations


class ImdbDataset(Dataset):
    def __init__(self, source_root, label_nc=19, split=(0.0, 0.9)):
        image_paths = sorted(glob.glob(os.path.join(source_root, '**/*.jpg')))
        self.image_paths = image_paths[int(len(image_paths) * split[0]):int(len(image_paths) * split[1])]
        with open(os.path.join(source_root, 'landmarks.pkl'), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.label_nc = label_nc
        self.to_one_hot = augmentations.ToOneHot(n_classes=self.label_nc)
        self.size = 256

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def resize(cls, image, size):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    @classmethod
    def transform(cls, img, size=256, to_segments=False, flip=True, norm=True):
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

    def forehead_coords(self, p):
        x0 = (p[0][0] + p[17][0]) / 2
        x1 = (p[16][0] + p[26][0]) / 2
        # y_min_i = np.argmin(p[17:27, 1])
        y0 = np.max(p[17:27, 1])
        # brows_sorted = sorted(p[17:27, 1].tolist())
        # y0 = brows_sorted[3]
        height = np.max(p[27:36, 1]) - np.min(p[0:17, 1])
        return np.array([x0, max(y0 - height, 0), x1, y0])

    def __getitem__(self, index):
        from_path = self.image_paths[index]
        from_im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGB)
        src_path = '/'.join(from_path.split('/')[-2:])
        landmark = self.denorm_lmarks(self.landmarks[src_path], from_im)
        fbox = self.forehead_coords(landmark).astype(int)
        if fbox[3] - fbox[1] < 16 or fbox[2] - fbox[0] < 16:
            # print(f'Fbox too small: {fbox}')
            new_idx = np.random.randint(0, len(self))
            return self[new_idx]

        fbox = fbox.astype(np.int32)
        forehead = from_im[fbox[1]:fbox[3],fbox[0]:fbox[2],:]
        forehead_sized = self.transform(forehead, self.size)
        forehead_sized = torch.tensor(forehead_sized).permute([2, 0, 1])
        code = np.random.randn(18,512).astype('float32')
        return torch.from_numpy(code),forehead_sized,torch.from_numpy(fbox)
