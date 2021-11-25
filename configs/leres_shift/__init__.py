import torch.nn as nn
import torch.optim as optim

from datasets.nyudepth_v2_shift import NYUDepthV2
from meters.leres import MeterLeresPCM
from evaluate.leres.eval import evaluate
from utils.config import Config, configs

# dataset configs
configs.dataset = Config(NYUDepthV2)
configs.dataset.root = 'data/nyudepth_v2'
configs.dataset.num_points = 2048

# evaluate configs
configs.evaluate = Config()
configs.evaluate.fn = evaluate
configs.evaluate.dataset = Config(split='test')

# train configs
configs.train = Config()
configs.train.num_epochs = 250
configs.train.batch_size = 8

# train: meters
configs.train.meters = Config()
configs.train.meters['acc/loss_{}'] = Config(MeterLeresPCM)

# train: metric for save best checkpoint
configs.train.metric = 'acc/loss_val'

# train: criterion
configs.train.criterion = Config(nn.MSELoss)

# train: optimizer
configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
