from datasets.sunrgbd import SunRGBD
from utils.config import Config, configs

configs.dataset = Config(SunRGBD)
configs.dataset.root = 'data/SUNRGBD'
configs.dataset.num_points = 2048
