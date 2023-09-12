import cv2
from module.builder import CLASSIFIER_PREPROCESS
import numpy as np
from PIL import Image
from module.logger_manager import get_root_logger


class ImageBase:
    def __init__(self, width, height, rgb=False):
        self.logger = get_root_logger()
        self.width = width
        self.height = height
        self.rgb = rgb


@CLASSIFIER_PREPROCESS.register_module
class RegularResize(ImageBase):

    def process(self, img):
        """
        :param img: 图片通过opencv读取， np.array()
        :return: cxhxw np.array()
        """
        img = cv2.resize(img, (self.width, self.height),
                         interpolation=cv2.INTER_LINEAR)
        img = (img - 127.5) / 127.5
        img = img.transpose((2, 0, 1))  # HWC to CHW, BGR
        if self.rgb:
            img = img[::-1]  # RGB
        img = np.ascontiguousarray(img)
        return img

    # Dataloader 通过路径对图片进行预处理
    def __call__(self, img_path):
        img = cv2.imread(img_path)
        self.process(img)
        return img


@CLASSIFIER_PREPROCESS.register_module
class SmokePhoneCrop(ImageBase):
    def process(self, img):
        """
        :param img: 图片通过opencv读取， np.array()
        :return: cxhxw np.array()
        """
        height, width, _ = img.shape
        h_, w_ = height, width

        offset = 1.
        offset_h = int((h_ * offset) // 10)
        offset_w = 0
        t1 = 5
        t = 6
        t0 = 8
        flag = 1

        # HPC settings
        if h_ > 2.5 * w_ and flag == 1:
            new_h = (h_ * t1) // 10
            img = img[offset_h:new_h, 0:width]
        elif h_ > 1.8 * w_ and flag == 1:
            new_h = (h_ * t) // 10
            img = img[offset_h:new_h, offset_w:(width-offset_w)]
        elif h_ > 1.5 * w_ and flag == 1:
            new_h = (h_ * t0) // 10
            img = img[offset_h:new_h, offset_w:(width-offset_w)]
        img = cv2.resize(img, (self.width, self.height))
        img = img.transpose((2, 0, 1))
        if self.rgb:
            img = img[::-1]  # RGB
        img = np.ascontiguousarray(img)
        img = img / 255.0
        return img

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = self.process(img)
        return img

