"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import logging
import torch
import numpy as np
import math
import cv2
from options.train_options import TrainOptions
from models.psp import pSp
import transform3d.euler as euler
import torchvision.transforms.functional as F


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    def draw_curve(img, idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        if img is None:
            return None
        for i in idx_list:
            img = cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            img = cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)
        return img


    img = draw_curve(img,list(range(0, 16)), color=(255, 144, 25))  # jaw
    img = draw_curve(img,list(range(17, 21)), color=(50, 205, 50))  # eye brow
    img = draw_curve(img,list(range(22, 26)), color=(50, 205, 50))
    img = draw_curve(img,list(range(27, 35)), color=(208, 224, 63))  # nose
    img = draw_curve(img,list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    img = draw_curve(img,list(range(42, 47)), loop=True, color=(71, 99, 255))
    img = draw_curve(img,list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    img = draw_curve(img,list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img

def tensor2im(var):
    var = var.cpu().numpy()
    var = np.transpose(var, [1, 2, 0])
    var = ((var + 1) / 2)
    var = np.clip(var,0,1)
    var = var * 255
    return var.astype(np.uint8)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    opts = TrainOptions()
    opts.parser.add_argument('--res_video', type=str, help='Result video')
    opts = opts.parse()
    
    if torch.cuda.is_available():
        device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
    else:
        device = 'cpu'

    opts.device = device

    net = pSp(opts).to(device)
    net.eval()

    data_file = os.path.join(opts.dataset_type, 'data.npz')
    npzfile = np.load(data_file)
    lands = npzfile['landmarks3']
    boxes = npzfile['boxes']
    f = 0
    land = lands[f].copy()
    box = boxes[f]
    kf = 3
    x1 = max(0, box[0] - (box[2] - box[0]) / kf)
    x2 = min(1, box[2] + (box[2] - box[0]) / kf)
    y1 = max(0, box[1] - (box[3] - box[1]) / kf)
    y2 = min(1, box[3] + (box[3] - box[1]) / kf)
    land[:, 0] = (land[:, 0] - x1) / (x2 - x1)
    land[:, 1] = (land[:, 1] - y1) / (y2 - y1)
    land[:,0:2] = np.clip(land[:,0:2], 0, 1)

    def _make_frame(oa1,oa2,oa3):
        a1 = oa1 * math.pi / 180
        a2 = oa2 * math.pi / 180
        a3 = oa3 * math.pi / 180

        m = euler.euler2mat(a1, a2, a3)
        nl = land.copy()
        nl = np.dot(m, nl.T).T
        nl[:, 0:2] = np.clip(nl[:, 0:2], 0, 1)
        nl *= 256
        nl = nl.astype(np.int32)
        nl = vis_landmark_on_img(np.ones((256,256, 3), dtype=np.uint8) * 255, nl)
        nl = F.to_tensor(nl)
        nl = F.normalize(nl, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return nl

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vout = cv2.VideoWriter(opts.res_video, fourcc, fps, (256,256))

    for a1 in range(50):
        a1 -= 25.0
        x = _make_frame(a1,0,0)
        with torch.no_grad():
            x = x.unsqueeze(0)
            x = x.to(device).float()
            y_hat, latent = net.forward(x, return_latents=True)
            y = tensor2im(y_hat[0])
        vout.write(y[:,:,::-1])
    for a1 in range(70):
        a1 -= 35.0
        x = _make_frame(0,a1,0)
        with torch.no_grad():
            x = x.unsqueeze(0)
            x = x.to(device).float()
            y_hat, latent = net.forward(x, return_latents=True)
            y = tensor2im(y_hat[0])
        vout.write(y[:,:,::-1])
    for a1 in range(50):
        a1 -= 25.0
        x = _make_frame(0,0,a1)
        with torch.no_grad():
            x = x.unsqueeze(0)
            x = x.to(device).float()
            y_hat, latent = net.forward(x, return_latents=True)
            y = tensor2im(y_hat[0])
        vout.write(y[:,:,::-1])
    vout.release()

        



if __name__ == '__main__':
    main()
