
import os
import cv2
import torch
import numpy as np
from module.builder import DETECTION_PREPROCESS


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


@DETECTION_PREPROCESS.register_module
class YoloProcess:
    def __init__(self, img_size, rgb=False):
        self.img_size = img_size
        self.rgb = rgb

    def load_image(self, image_path):
        im = cv2.imread(image_path)  # BGR
        if im is None:
            self.logger.error(f"Image is None, {image_path}, {os.path.exists(image_path)}")
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def __call__(self, image_path):
        # Load image
        img, (h0, w0), (h, w) = self.load_image(image_path)

        # Letterbox
        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=False)
        shapes = torch.Tensor((h0, w0))

        # Convert
        if self.rgb:
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        else:
            img = img.transpose((2, 0, 1))  # HWC to CHW, BGR
        img = np.ascontiguousarray(img)
        img = img/255.0
        return img, shapes

