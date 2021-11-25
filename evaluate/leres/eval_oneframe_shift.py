import argparse
import os
import sys
import cv2
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--focal_configs', nargs='+')
parser.add_argument('--shift_configs', nargs='+')
parser.add_argument('--devices', default=None)
parser.add_argument('--depth_fpath', default=None)
args, opts = parser.parse_known_args()

def prepare(configs_path):
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {configs_path}')
    configs.update_from_modules(*configs_path)

    # define save path
    save_path = get_save_path(*configs_path, prefix='runs')
    configs.train.save_path = save_path
    configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
    configs.train.best_checkpoint_path = os.path.join(save_path, 'best.pth.tar')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus
    if 'best_checkpoint_path' not in configs.evaluate or configs.evaluate.best_checkpoint_path is None:
        if 'best_checkpoint_path' in configs.train and configs.train.best_checkpoint_path is not None:
            configs.evaluate.best_checkpoint_path = configs.train.best_checkpoint_path
        else:
            configs.evaluate.best_checkpoint_path = os.path.join(configs.train.save_path, 'best.pth.tar')
    assert configs.evaluate.best_checkpoint_path.endswith('.pth.tar')
    configs.evaluate.stats_path = configs.evaluate.best_checkpoint_path.replace('.pth.tar', '.eval.npy')

    return copy.copy(configs)


def build_intrinsics(focal_len, cx, cy):
    K = np.zeros((3, 3), dtype=np.float32)
    K[0][0] = focal_len
    K[1][1] = focal_len
    K[0][2] = cx
    K[1][2] = cy
    K[2][2] = 1
    return K


def load_depth_img(depth_fname):
    inv_depth_raw = cv2.imread(depth_fname, -1)
    valid_mask = (inv_depth_raw < 65287)

    depth = (351.3 / (1092.5 - inv_depth_raw)).astype(np.float32)
    shift = -depth.min() + 0.5
    depth_norm = depth + shift
    dmax = np.percentile(depth_norm, 98)
    depth_norm = depth_norm / dmax
    shift = shift / dmax
    print("shift {}".format(shift))

    return depth_norm, valid_mask


def depth_to_ptcloud(depth, valid_mask, width, height, Kinv):
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

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
    ptcloud[3, :, :] = (xv.astype(np.float32) - width / 2)  / width
    ptcloud[4, :, :] = (yv.astype(np.float32) - height / 2) / width
    ptcloud = ptcloud.reshape(5, xv.shape[0] * xv.shape[1])
    valid_mask = valid_mask.reshape(xv.shape[0] * xv.shape[1])
    valid_indices = np.where(valid_mask)

    return ptcloud[:, valid_indices[0]]


def leres_find_focal(depth, valid_mask, focal_model, device, focal_len=None):
    width = depth.shape[1]
    height = depth.shape[0]

    if focal_len == None:
        focal_len = 0.9 * width
    K = build_intrinsics(focal_len, width / 2.0, height / 2.0)
    Kinv = np.linalg.inv(K)

    point_set = depth_to_ptcloud(depth, valid_mask, width, height, Kinv)
    choice = np.random.choice(point_set.shape[1], 2048, replace=True)
    point_set = point_set[:, choice]

    # model inference
    inputs = torch.from_numpy(
        point_set.reshape(1, point_set.shape[0], point_set.shape[1])
    ).float().to(device)

    with torch.no_grad():
        prediction = focal_model(inputs)
        focal_scale = prediction.item()
        proposed_focal_len = focal_len / focal_scale
        print("orig focal {} focal length scale {} proposed_focal_len {}".
              format(focal_len, focal_scale, proposed_focal_len))

    return proposed_focal_len


def leres_find_shift(depth, valid_mask, shift_model, device, focal_len):
    width = depth.shape[1]
    height = depth.shape[0]

    K = build_intrinsics(focal_len, width / 2.0, height / 2.0)
    Kinv = np.linalg.inv(K)

    point_set = depth_to_ptcloud(depth, valid_mask, width, height, Kinv)
    choice = np.random.choice(point_set.shape[1], 2048, replace=True)
    point_set = point_set[:, choice]

    # model inference
    inputs = torch.from_numpy(
        point_set.reshape(1, point_set.shape[0], point_set.shape[1])
    ).float().to(device)

    with torch.no_grad():
        prediction = shift_model(inputs)
        depth_shift = prediction.item()
        print("depth shift {}".format(depth_shift))
        leres_depth = depth - depth_shift

    return leres_depth


def evaluate():
    focal_configs = prepare(args.focal_configs)
    shift_configs = prepare(args.shift_configs)

    ###########
    # Prepare #
    ###########
    print(focal_configs)
    print(shift_configs)

    if focal_configs.device == 'cuda':
        cudnn.benchmark = True
        if focal_configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False

    #################################
    # Initialize DataLoaders, Model #
    #################################
    print(f'\n==> creating model "{focal_configs.model}"')
    focal_model = focal_configs.model()
    if focal_configs.device == 'cuda':
        focal_model = torch.nn.DataParallel(focal_model)
    focal_model = focal_model.to(focal_configs.device)

    if os.path.exists(focal_configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{focal_configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(focal_configs.evaluate.best_checkpoint_path)
        focal_model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    focal_model.eval()

    print(f'\n==> creating model "{shift_configs.model}"')
    shift_model = shift_configs.model()
    if shift_configs.device == 'cuda':
        shift_model = torch.nn.DataParallel(shift_model)
    shift_model = shift_model.to(shift_configs.device)

    if os.path.exists(shift_configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{shift_configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(shift_configs.evaluate.best_checkpoint_path)
        shift_model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    shift_model.eval()

    ##############
    # Evaluation #
    ##############
    depth, valid_mask = load_depth_img(args.depth_fpath)
    proposed_focal_len = leres_find_focal(depth, valid_mask, focal_model, focal_configs.device)
    proposed_depth = leres_find_shift(depth, valid_mask, focal_model,
                                      shift_configs.device, proposed_focal_len)

    dmax = np.percentile(proposed_depth, 98)
    depth_norm = proposed_depth / dmax
    proposed_focal_len = leres_find_focal(depth_norm, valid_mask, focal_model,
                                          focal_configs.device, focal_len=proposed_focal_len)


if __name__ == '__main__':
    evaluate()
