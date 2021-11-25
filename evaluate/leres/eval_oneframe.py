import argparse
import os
import random
import sys
import cv2
import numpy as np

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('configs', nargs='+')
parser.add_argument('--devices', default=None)
parser.add_argument('--depth_fpath', default=None)
args, opts = parser.parse_known_args()

def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)

    # define save path
    save_path = get_save_path(*args.configs, prefix='runs')
    os.makedirs(save_path, exist_ok=True)
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

    return configs


def load_depth(depth_fname):
    depth = np.load(depth_fname)
    valid_mask = np.ones(depth.shape, dtype=bool)

    depth_norm = depth - depth.min() + 0.5
    dmax = np.percentile(depth_norm, 98)
    depth_norm = depth_norm / dmax

    return depth_norm, valid_mask


def load_depth_img(depth_fname):
    inv_depth_raw = cv2.imread(depth_fname, -1)
    valid_mask = (inv_depth_raw < 65287)

    depth = (351.3 / (1092.5 - inv_depth_raw)).astype(np.float32)
    depth_norm = depth - depth.min() + 0.5
    dmax = np.percentile(depth_norm, 98)
    depth_norm = depth_norm / dmax

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


def evaluate(configs=None):
    configs = prepare() if configs is None else configs

    import torch
    import torch.backends.cudnn as cudnn

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False

    print(configs)

    #################################
    # Initialize DataLoaders, Model #
    #################################
    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    if configs.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(configs.device)

    if os.path.exists(configs.evaluate.best_checkpoint_path):
        print(f'==> loading checkpoint "{configs.evaluate.best_checkpoint_path}"')
        checkpoint = torch.load(configs.evaluate.best_checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    elif os.path.exists(configs.train.checkpoint_path):
        print(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path)
        model.load_state_dict(checkpoint.pop('model'))
        del checkpoint
    else:
        return

    model.eval()

    ##############
    # Evaluation #
    ##############
    #depth, valid_mask = load_depth_img(args.depth_fpath)
    depth, valid_mask = load_depth(args.depth_fpath)

    width = depth.shape[1]
    height = depth.shape[0]
    K = np.zeros((3, 3), dtype=np.float32)
    seed = np.random.rand()

    focal_len_scale = seed * (1.25 - 0.6) + 0.6
    print("focal_len_scale {}".format(focal_len_scale))
    focal_len = focal_len_scale * 570
    K[0][0] = focal_len
    K[1][1] = focal_len
    K[0][2] = width / 2.0
    K[1][2] = height / 2.0
    K[2][2] = 1
    Kinv = np.linalg.inv(K)

    point_set = depth_to_ptcloud(depth, valid_mask, width, height, Kinv)
    choice = np.random.choice(point_set.shape[1], 2048, replace=True)
    point_set = point_set[:, choice]

    # model inference
    inputs = torch.from_numpy(
        point_set.reshape(1, point_set.shape[0], point_set.shape[1])
    ).float().to(configs.device)

    prediction = 0
    with torch.no_grad():
        prediction = model(inputs)
        print("prediction {}".format(prediction))


if __name__ == '__main__':
    evaluate()
