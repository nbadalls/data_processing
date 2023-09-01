import torch
from torch.utils.data import Dataset
from module.logger_manager import get_root_logger
from module.builder import LOADER
from ..registry import build_from_cfg
from ..builder import DETECTION_PREPROCESS, CLASSIFIER_PREPROCESS


class BaseLoader(Dataset):
    def __init__(self, image_list):
        self.img_files = image_list  # update
        self.logger = get_root_logger()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        pass


@LOADER.register_module
class LoadImagesYolo(BaseLoader):
    def __init__(self, image_list, preprocess):
        super().__init__(image_list)
        self.preprocess = build_from_cfg(preprocess, DETECTION_PREPROCESS)

    def __getitem__(self, index):
        # Load image
        img, shapes = self.preprocess(self.img_files[index])
        return torch.from_numpy(img).float(), self.img_files[index], shapes


@LOADER.register_module
class LoadImageCls(BaseLoader):
    def __init__(self, image_list, preprocess):
        super().__init__(image_list)
        self.preprocess = build_from_cfg(preprocess, CLASSIFIER_PREPROCESS)

    def __getitem__(self, index):
        img = self.preprocess(self.img_files[index])
        return torch.from_numpy(img).float(), self.img_files[index]


