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


@CLASSIFIER_PREPROCESS.register_module
class ImageRatioCrop:
    def __init__(self, img_hw_ratio, h_top_crop, h_bottom_crop, w_left_crop, w_right_crop):
        self.img_hw_ratio = img_hw_ratio
        self.h_top = h_top_crop
        self.h_bottom = h_bottom_crop
        self.w_left = w_left_crop
        self.w_right = w_right_crop

    def process(self, image, bbox=None):
        if bbox is None:
            src_x1, src_y1, src_x2, src_y2 = 0, 0, image.shape[1]-1, image.shape[0]-1
        else:
            src_x1, src_y1, src_x2, src_y2 = bbox
        img_h, img_w = image.shape[0], image.shape[1]
        bbox_w, bbox_h = src_x2 - src_x1, src_y2 - src_y1
        ratio = bbox_h / bbox_w
        diff = ratio - np.array(self.img_hw_ratio)
        if max(diff) >= 0:
            i = np.argmin(diff[np.where(diff >= 0)])
            x1 = src_x1 + int(bbox_w * self.w_left[i])
            y1 = src_y1 + int(bbox_h * self.h_top[i])
            x2 = src_x2 - int(bbox_w * self.w_right[i])
            y2 = src_y2 - int(bbox_h * self.h_bottom[i])

            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1)
            crop_img = image[y1:y2 + 1, x1:x2 + 1]
        else:
            crop_img = image[src_y1:src_y2 + 1, src_x1:src_x2 + 1]
        return crop_img

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = self.process(img)
        return img

