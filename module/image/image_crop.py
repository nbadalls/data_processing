import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from ..utility import make_if_not_exists
from ..logger_manager import get_root_logger
from itertools import repeat
from ..builder import CROPPER


@CROPPER.register_module
class CropImage:
    def __init__(self, pool_num=10, scale_w=1.0, scale_h=1.0):
        """
        :param pool_num:  多进程操作的进程数量
        :param scale_w:  w扩展为原来的多少倍
        :param scale_h:  h扩展为原来的多少倍
        """
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.pool_num = pool_num
        self.logger = get_root_logger()
        self.logger.info(f"截取参数 scale_w: {scale_w}  scale_h: {scale_h}")

    def crop_image(self, image_files, rect_rets, image_root_folder):
        """
        :param image_files: 截取图片的全路径
        :param rect_rets: 检测目标的结果 [[x1,y1,x2,y2,conf,label], [x1,y1,x2,y2,conf,label]]
        :param image_root_folder: 根路径用于生成截取之后图片的路径 "{image_root_folder}_crop_scale-w_{scale_w}_scale-h_{scale_h}"
        """
        image_crop_folder = f"{image_root_folder}_crop_sw_{self.scale_w}_sh_{self.scale_h}"
        if os.path.exists(image_crop_folder):
            crop_files = [f"{image_crop_folder}/{elem}" for elem in os.listdir(image_crop_folder)]
            self.logger.info(f"已存在文件夹{image_crop_folder}，共有图片{len(crop_files)}张")
            return crop_files

        self.logger.info(f"待截取图片{len(image_files)}张  存放路径为{image_crop_folder}")
        make_if_not_exists(image_crop_folder)
        sum_num = 0
        crop_files = []
        with Pool(self.pool_num) as p:
            par = tqdm(p.imap_unordered(self.cropper, zip(repeat(image_crop_folder), image_files,
                                                          rect_rets, repeat(self.logger),
                                                          repeat(self.scale_w), repeat(self.scale_h))),
                       total=len(image_files))
            for img_path in par:
                crop_files += img_path
                sum_num += len(img_path)
        self.logger.info(f"共截取图片{len(crop_files)}张")
        return crop_files

    @staticmethod
    def cropper(args):
        # bbox
        image_crop_folder, image_path, rect_rets, logger, scale_w, scale_h = args
        crop_files = []
        if len(rect_rets) > 0:
            img = cv2.imread(image_path)
            image_name = Path(image_path).stem
            img_h, img_w, _ = img.shape
            if img is None:
                logger.error(f"无法打开图片：\n{image_path}")
                return 0

            xyxy, label = np.array(rect_rets)[:, :4], np.array(rect_rets)[:, 5]
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            deta_x, deta_y = (scale_w*w-w)//2,  (scale_h*h-h)//2
            deta_xy = np.stack((-deta_x, -deta_y, deta_x, deta_y), axis=-1)
            xyxy += deta_xy
            xyxy[:, :2] = np.maximum(xyxy[:, :2], np.array([0, 0]))
            xyxy[:, 2:4] = np.minimum(xyxy[:, 2:4], np.array([img_w-1, img_h-1]))

            for i in range(xyxy.shape[0]):
                bbox = xyxy[i, :4].astype(np.int)
                img_crop = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
                crop_img_name = f"{image_name}_{str(i).zfill(2)}_l-{int(label[i])}.jpg"
                crop_img_path = f"{image_crop_folder}/{crop_img_name}"
                crop_files.append(crop_img_path)
                cv2.imwrite(crop_img_path, img_crop)
        return crop_files





