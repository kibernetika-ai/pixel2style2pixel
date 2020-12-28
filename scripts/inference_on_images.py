import argparse
import glob
import os

import cv2
import face_alignment
import numpy as np
import torch

from datasets import imdb_dataset
from models.psp import pSp
from scripts import align_all_parallel
from utils.common import tensor2im


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output-dir')

    return parser.parse_args()


def main():
    args = parse_args()
    paths = glob.glob(os.path.join(args.image_dir, '*.jpg'))

    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts = argparse.Namespace(**opts)
    opts.checkpoint_path = args.checkpoint_path
    opts.decoder_train_disable = False
    net = pSp(opts)
    net.eval()
    if torch.cuda.is_available():
        net.cuda()

    print('Load face detect driver...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device)
    print('Done loading.')

    for path in paths:
        input_orig = cv2.imread(path)
        input_rgb = cv2.cvtColor(input_orig, cv2.COLOR_BGR2RGB)

        l3d = fa3d.get_landmarks(input_rgb)
        aligned = align_all_parallel.align_face(None, None, img=input_rgb, landmarks=l3d[0][:, :2])

        resized = cv2.resize(np.array(aligned), (256, 256))
        inputs = imdb_dataset.ImdbDataset.transform(resized, flip=False, norm=True)
        inputs = torch.tensor(inputs).permute([2, 0, 1]).unsqueeze(0)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        result_batch = net(inputs)
        result = tensor2im(result_batch[0])

        if args.show:
            cv2.imshow('Res', np.hstack((resized[:, :, ::-1], np.array(result)[:, :, ::-1])))
            cv2.waitKey(0)

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_path = os.path.join(args.output_dir, os.path.basename(path))
            cv2.imwrite(save_path, np.array(result)[:, :, ::-1])


if __name__ == '__main__':
    main()



