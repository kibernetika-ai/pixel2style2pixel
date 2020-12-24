import glob
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImdbDataset(Dataset):
    def __init__(self, source_root, source_transform=None, split=(0.0, 0.9)):
        image_paths = sorted(glob.glob(os.path.join(source_root, '**/*.jpg')))
        self.image_paths = image_paths[int(len(image_paths) * split[0]):int(len(image_paths) * split[1])]
        with open(os.path.join(source_root, 'landmarks.pkl'), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.source_transform = source_transform
        self.size = 256

    def __len__(self):
        return len(self.image_paths)

    def resize(self, image):
        return cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def transform(self, img):
        img = self.resize(img)
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)  # flip horizontally

        img = img.astype(np.float32) / 255.
        img = (img - 0.5) / 0.5
        return img

    def inject_forehead(self, src_path, dest_path, src, dest):
        src_path = '/'.join(src_path.split('/')[-2:])
        dest_path = '/'.join(dest_path.split('/')[-2:])
        landmark = self.denorm_lmarks(self.landmarks[src_path], src)
        landmark_dst = self.denorm_lmarks(self.landmarks[dest_path], dest)
        forehead_coords = self.forehead_coords(landmark)
        forehead_coords_dst = self.forehead_coords(landmark_dst)

        injected = self.insert_forehead(dest, src, forehead_coords, forehead_coords_dst)
        return injected

    def insert_forehead(self, var, src=None, roi=None, roi_dest=None):
        r = var.astype('uint8')
        roi = roi.astype(np.int)
        roi_dest = roi_dest.astype(np.int)
        src_forehead = src[roi[1]:roi[3], roi[0]:roi[2]]
        dest_forehead = cv2.resize(src_forehead, (roi_dest[2] - roi_dest[0], roi_dest[3] - roi_dest[1]), cv2.INTER_AREA)
        center = ((roi_dest[0] + roi_dest[2]) // 2, (roi_dest[1] + roi_dest[3]) // 2)
        r = cv2.seamlessClone(dest_forehead, r, np.ones_like(dest_forehead) * 255, center, cv2.MIXED_CLONE)
        # r[roi_dest[1]:roi_dest[3], roi_dest[0]:roi_dest[2], :] = dest_forehead
        return r

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

        to_path = np.random.choice(self.image_paths)
        while to_path == from_path:
            to_path = np.random.choice(self.image_paths)

        to_im = cv2.cvtColor(cv2.imread(to_path), cv2.COLOR_BGR2RGB)
        to_im_fhead = self.inject_forehead(from_path, to_path, from_im, to_im)

        # src_path = '/'.join(from_path.split('/')[-2:])
        # dest_path = '/'.join(to_path.split('/')[-2:])
        # landmark = self.denorm_lmarks(self.landmarks[src_path], from_im)
        # landmark_dst = self.denorm_lmarks(self.landmarks[dest_path], to_im)
        # for p in landmark_dst:
        #     cv2.circle(
        #         to_im_fhead,
        #         (p[0], p[1]),
        #         2,
        #         (0, 250, 0)
        #     )

        # cv2.imshow('img1', from_im[:, :, ::-1])
        # cv2.imshow('img2', to_im[:, :, ::-1])
        # cv2.imshow('img3', to_im_fhead[:, :, ::-1])
        # cv2.waitKey(0)
        # exit()
        from_im = self.transform(from_im)
        to_im_fhead = self.transform(to_im_fhead)

        return torch.tensor(from_im), torch.tensor(to_im_fhead)
