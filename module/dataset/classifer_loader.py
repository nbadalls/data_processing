import os
import cv2
import torch
from torch.utils.data import Dataset
from module.logger_manager import get_root_logger
from module.builder import LOADER
from ..registry import build_from_cfg
from ..builder import CLASSIFIER_PREPROCESS


@LOADER.register_module
class LoadImageCls(Dataset):
    def __init__(self, image_list, preprocess):
        """
        :param image_list: 输入图像的路径的list或者字符串
        :param preprocess:   处理输入图像的输入参数
        """
        self.logger = get_root_logger()
        if isinstance(image_list, str):
            image_path = []
            for root_path, folder_path, file_path in os.walk(image_list):
                for elem in file_path:
                    image_path.append(f"{root_path}/{elem}")
            self.img_files = image_path
        elif isinstance(image_list, list):
            self.img_files = image_list
        else:
            self.img_files = None
            self.logger.error(f"image_list的类型是str或者list，现在是{type(image_list)}")

        self.logger = get_root_logger()
        self.preprocess = build_from_cfg(preprocess, CLASSIFIER_PREPROCESS)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.preprocess is not None:
            img = self.preprocess(self.img_files[index])
        return torch.from_numpy(img).float(), self.img_files[index]