
import numpy as np
import torch
from torch.utils.data import Dataset
from module.logger_manager import get_root_logger
from module.builder import LOADER
from ..registry import build_from_cfg
from ..builder import DETECTION_PREPROCESS


@LOADER.register_module
class LoadImagesYolo(Dataset):
    def __init__(self, image_list, preprocess, img_size=640):

        self.img_size = img_size
        self.img_files = image_list  # update
        self.logger = get_root_logger()
        self.preprocess = build_from_cfg(preprocess, DETECTION_PREPROCESS)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load image
        img, shapes = self.preprocess(self.img_files[index])
        return torch.from_numpy(img).float(), self.img_files[index], shapes


