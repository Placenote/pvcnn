import json
import os
import cv2

import numpy as np
from torch.utils.data import Dataset

__all__ = ['NYUDepthV2']


class _NYUDepthV2Dataset(Dataset):
    def __init__(self, root, num_points, split='train'):
        self.root = root
        self.num_points = num_points
        self.split = split
        self.focal_len = 582.62448
        self.cx = 313.04475870804731
        self.cy = 238.44389626620386
        self.width = 640
        self.height = 480

        self.K = self.build_intrinsics(self.focal_len, self.cx, self.cy)
        self.Kinv = np.linalg.inv(self.K)

        file_list = []
        with open(os.path.join(self.root, '{}.txt'.format(split)), 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                if ".pgm" not in lines[i]:
                    continue
                depth_fname = os.path.join(self.root, lines[i].strip())
                file_list.append(depth_fname)

        self.file_paths = file_list

        x = np.arange(0, self.width, 1)
        y = np.arange(0, self.height, 1)
        self.uv, self.vv = np.meshgrid(x, y, sparse=False, indexing='xy')

    def __getitem__(self, index):
        focal_len_scale = np.random.rand() * (1.25 - 0.6) + 0.6
        depth_fname = self.file_paths[index]
        depth, valid_mask = self.load_depth(depth_fname)

        # generate distorted pointset based on perturned focal_len
        perturbed_focal_len = self.focal_len * focal_len_scale
        K = self.build_intrinsics(perturbed_focal_len, self.cx, self.cy)
        Kinv = np.linalg.inv(K)

        point_set = self.depth_to_ptcloud(depth, valid_mask, self.width,
            self.height, Kinv, self.uv, self.vv)

        choice = np.random.choice(point_set.shape[1], self.num_points, replace=True)
        point_set = point_set[:, choice]
        return point_set, focal_len_scale

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def build_intrinsics(focal_len, cx, cy):
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = focal_len
        K[1][1] = focal_len
        K[0][2] = cx
        K[1][2] = cy
        K[2][2] = 1
        return K

    def load_depth(self, depth_fname):
        inv_depth_raw = cv2.imread(depth_fname, -1)
        if inv_depth_raw is None:
            print("{} does not exist!?".format(depth_fname))
        valid_mask = (inv_depth_raw < 65287)

        depth = (351.3 / (1092.5 - inv_depth_raw)).astype(np.float32)
        return depth, valid_mask

    @staticmethod
    def depth_to_ptcloud(depth, valid_mask, width, height, Kinv, xv, yv):
        homo_coords = np.zeros((3, xv.shape[0], xv.shape[1]), dtype="float32")
        homo_coords[0, :, :] = xv
        homo_coords[1, :, :] = yv
        homo_coords[2, :, :] = np.ones(xv.shape)
        homo_coords = homo_coords.reshape((3, xv.shape[0] * xv.shape[1]))

        rays = Kinv.dot(homo_coords)
        rays = rays.reshape((3, xv.shape[0], xv.shape[1]))
        ptcloud = np.zeros((5, xv.shape[0], xv.shape[1]), dtype="float32")

        ptcloud[0, :, :] = np.multiply(rays[0, :, :], depth)
        ptcloud[1, :, :] = np.multiply(rays[1, :, :], depth)
        ptcloud[2, :, :] = np.multiply(rays[2, :, :], depth)
        ptcloud[3, :, :] = (xv - width / 2) / width
        ptcloud[4, :, :] = (yv - height / 2) / width
        ptcloud = ptcloud.reshape(5, xv.shape[0] * xv.shape[1])
        valid_mask = valid_mask.reshape(xv.shape[0] * xv.shape[1])
        valid_indices = np.where(valid_mask)

        return ptcloud[:, valid_indices[0]]

class NYUDepthV2(dict):
    def __init__(self, root, num_points):
        super().__init__()
        self['train'] = _NYUDepthV2Dataset(root=root, split='train', num_points=num_points)
        self['val'] = _NYUDepthV2Dataset(root=root, split='val', num_points=num_points)
