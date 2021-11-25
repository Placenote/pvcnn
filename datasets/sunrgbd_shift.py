import json
import os
import cv2

import numpy as np
from torch.utils.data import Dataset

__all__ = ['SunRGBD']


class _SunRGBDDataset(Dataset):
    def __init__(self, root, num_points, split='train'):
        self.root = root
        self.num_points = num_points
        self.split = split

        file_list = []
        intrins_file_list = []
        with open(os.path.join(self.root, '{}.txt'.format(split)), 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                depth_fname = os.path.join(self.root, lines[i].strip())
                img_folder = os.path.dirname(depth_fname)
                intrins_fname = os.path.join(img_folder, "../intrinsics.txt")
                file_list.append(depth_fname)
                intrins_file_list.append(intrins_fname)

        self.file_paths = file_list
        self.intrins_paths = intrins_file_list


    def __getitem__(self, index):
        depth_fname = self.file_paths[index]
        intrins_fname = self.intrins_paths[index]
        depth, valid_mask, existing_shift = self.load_depth(depth_fname)

        focal_len, cx, cy = self.load_intrinsics(intrins_fname)
        width = depth.shape[1]
        height = depth.shape[0]
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)
        uv, vv = np.meshgrid(x, y, sparse=False, indexing='xy')

        # generate distorted pointset based on perturned focal_len
        K = self.build_intrinsics(focal_len, cx, cy)
        Kinv = np.linalg.inv(K)
        depth_shift = np.random.rand() * (0.8 + 0.25) - 0.25
        depth_fname = self.file_paths[index]

        # generate distorted pointset based on perturned shift
        shifted_depth = depth + depth_shift + existing_shift

        point_set = self.depth_to_ptcloud(shifted_depth, valid_mask, width,
            height, Kinv, uv, vv)

        choice = np.random.choice(point_set.shape[1], self.num_points, replace=True)
        point_set = point_set[:, choice]
        return point_set, depth_shift

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
        depth_shift = -depth.min() + 0.5
        depth_norm = depth + depth_shift
        dmax = np.percentile(depth_norm, 98)
        depth_norm = depth_norm / dmax
        depth_shift = depth_shift / dmax

        return depth_norm, valid_mask, depth_shift

    def load_intrinsics(self, intrins_fname):
        intrins_mat = np.zeros((3, 3))
        if os.path.exists(intrins_fname):
            with open(intrins_fname, 'r') as f:
                lines = f.readlines()
                if len(lines) == 3:
                    intrins_mat = np.loadtxt(intrins_fname, comments="#", delimiter=" ")
                elif len(lines) == 1:
                    text = lines[0].rstrip()
                    intrins_mat = np.fromstring(text, sep=" ")
                    intrins_mat = intrins_mat.reshape(3, 3)
                focal_len = intrins_mat[0][0]
                cx = intrins_mat[0][2]
                cy = intrins_mat[1][2]
                return focal_len, cx, cy

        focal_len = 582
        cx = 320
        cy = 240
        return focal_len, cx, cy

    @staticmethod
    def depth_to_ptcloud(depth, valid_mask, width, height, Kinv, xv, yv):
        homo_coords = np.zeros((3, xv.shape[0], xv.shape[1]), dtype="float32")
        homo_coords[0, :, :] = xv
        homo_coords[1, :, :] = yv
        homo_coords[2, :, :] = np.ones(xv.shape)
        homo_coords = homo_coords.reshape((3, xv.shape[0] * xv.shape[1]))

        rays = Kinv.dot(homo_coords)
        rays = rays.reshape((3, xv.shape[0], xv.shape[1]))
        ptcloud = np.zeros((3, xv.shape[0], xv.shape[1]), dtype="float32")

        ptcloud[0, :, :] = np.multiply(rays[0, :, :], depth)
        ptcloud[1, :, :] = np.multiply(rays[1, :, :], depth)
        ptcloud[2, :, :] = np.multiply(rays[2, :, :], depth)
        ptcloud = ptcloud.reshape(3, xv.shape[0] * xv.shape[1])
        valid_mask = valid_mask.reshape(xv.shape[0] * xv.shape[1])
        valid_indices = np.where(valid_mask)

        return ptcloud[:, valid_indices[0]]

class SunRGBD(dict):
    def __init__(self, root, num_points):
        super().__init__()
        self['train'] = _SunRGBDDataset(root=root, split='train', num_points=num_points)
        self['val'] = _SunRGBDDataset(root=root, split='val', num_points=num_points)
