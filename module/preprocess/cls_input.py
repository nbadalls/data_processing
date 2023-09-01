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
    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.width, self.height),
                         interpolation=cv2.INTER_LINEAR)
        img = (img-127.5)/127.5
        img = img.transpose((2, 0, 1))  # HWC to CHW, BGR
        if self.rgb:
           img = img[::-1] #RGB
        img = np.ascontiguousarray(img)
        return img


@CLASSIFIER_PREPROCESS.register_module
class SmokePhoneCrop(ImageBase):
    def __call__(self, img_path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(img_path, 'rb') as f:
            img = Image.open(f)

            width = img.size[0]
            height = img.size[1]
            h_, w_ = height, width

            offset = 1.
            offset_h = (h_ * offset) // 10
            offset_w = 0
            t1 = 5
            t = 6
            t0 = 8
            flag = 1

            # HPC settings
            if h_ > 2.5 * w_ and flag == 1:
                new_h = (h_ * t1) // 10
                img = img.crop((0, offset_h, width, new_h))
            elif h_ > 1.8 * w_ and flag == 1:
                new_h = (h_ * t) // 10
                img = img.crop((0 + offset_w, offset_h, width - offset_w, new_h))
            elif h_ > 1.5 * w_ and flag == 1:
                new_h = (h_ * t0) // 10
                img = img.crop((0 + offset_w, offset_h, width - offset_w, new_h))
            img = img.resize((self.width, self.height))
            if not self.rgb:
                img = img.convert("BGR")
            img = np.array(img)
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = img / 255.0
        return img


