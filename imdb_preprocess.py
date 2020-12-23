import argparse
import glob
import math
import os
import pickle
import sys

import cv2
import face_alignment
import numpy as np
import torch

from transform3d import euler


face_model_path = (
    '/opt/intel/openvino/deployment_tools/intel_models'
    '/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def norm3d_t(landmark, ref):
    t, _, _ = best_fit_transform(landmark, ref)
    #print(t)
    n = np.dot(t[0:3, 0:3], landmark.T).T
    n += t[:3, 3]
    return n.astype(np.float32), t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--min-size', default=100, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    min_face_size = args.min_size
    min_box_diagonal = int(math.sqrt(2 * (min_face_size ** 2)))
    print_fun('List files...')
    image_paths = glob.glob(os.path.join(args.data_dir, '**/*.jpg'))
    print_fun(f'Done list files: {len(image_paths)}')

    print_fun('Load face detect driver...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device)
    print_fun('Done loading.')

    landmark_ref = np.load('./landmark_ref.npy')
    processed = 0
    threshold = 0.2

    landmarks = {}
    boxes = {}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print_fun(f'Progress {i / len(image_paths) * 100:.2f} %.')
            print_fun(f'Processed {processed} images, looked: {i}.')

        try:
            with open(path, 'rb') as f:
                raw_img = f.read()
            frame = cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = fa3d.face_detector.detect_from_image(frame_rgb)
        except Exception as e:
            print_fun(f'ERROR: {path}, {e}; Skip')
            continue

        if len(detected_faces) != 1:
            continue
        box = detected_faces[0]
        x1 = box[0] / frame.shape[1]
        x2 = box[2] / frame.shape[1]
        y1 = box[1] / frame.shape[0]
        y2 = box[3] / frame.shape[0]
        l3d = fa3d.get_landmarks(frame, detected_faces=[box])
        scale = (box[2] - box[0] + box[3] - box[1]) / 195
        if l3d is None or len(l3d) < 1:
            continue
        l3d = l3d[0]
        landmark_3d = l3d[:, :].astype(np.float32)
        landmark_3d[:, 0] = (landmark_3d[:, 0]) / frame.shape[1]
        landmark_3d[:, 1] = (landmark_3d[:, 1]) / frame.shape[0]
        landmark_3d[:, 2] = landmark_3d[:, 2] / (200 * scale)
        landmark_3d[:, 0:2] = np.clip(landmark_3d[:, 0:2], 0, 1)
        _, land_transform = norm3d_t(landmark_3d.copy(), landmark_ref)

        a1, a2, a3 = euler.mat2euler(land_transform)

        # print(a1, a2, a3)
        # cv2.imshow('Image', frame)
        # cv2.waitKey(0)

        is_frontal = abs(a1) < threshold and abs(a2) < threshold and abs(a3) < threshold
        if not is_frontal:
            continue

        dirname = path.split('/')[-2]
        basename = os.path.basename(path)
        save_path = os.path.join(args.output_dir, dirname, basename)
        if not os.path.exists(os.path.join(args.output_dir, dirname)):
            os.makedirs(os.path.join(args.output_dir, dirname))
        with open(save_path, 'wb') as f:
            f.write(raw_img)

        landmarks[f'{dirname}/{basename}'] = landmark_3d
        boxes[f'{dirname}/{basename}'] = np.array([x1, y1, x2, y2])
        processed += 1

        if args.limit != 0 and processed >= args.limit:
            break

    with open(os.path.join(args.output_dir, 'landmarks.pkl'), 'wb') as f:
        pickle.dump(landmarks, f)
    with open(os.path.join(args.output_dir, 'boxes.pkl'), 'wb') as f:
        pickle.dump(boxes, f)

    print_fun(f'Processed {processed} images, looked: {i}')


def print_fun(s):
    print(s)
    sys.stdout.flush()


def box_diagonal(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return math.sqrt(w ** 2 + h ** 2)


if __name__ == '__main__':
    main()
