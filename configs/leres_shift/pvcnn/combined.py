from datasets.sunrgbd_shift import SunRGBD
from utils.config import Config, configs

configs.dataset = Config(SunRGBD)
configs.dataset.root = 'data/combined'
configs.dataset.num_points = 2048
