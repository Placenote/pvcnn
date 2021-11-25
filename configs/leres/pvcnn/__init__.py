import torch.optim as optim

from models.leres import PVCNN
from utils.config import Config, configs

# model
configs.model = Config(PVCNN)
configs.model.extra_feature_channels = 2

configs.train.num_epochs = 250
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
