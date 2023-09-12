"""
模型处理数据的pipline
"""

import os, pickle, sys

import numpy as np

sys.path.append("./")
from module.registry import build_from_cfg
from module.builder import DETECTOR, CLASSIFIER, VIDEOPROCESS, CROPPER
from module.utility import get_full_image_path
from .base_pipline import PipBase


class Processor(PipBase):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        # 按照配置文件的顺序依次添加执行流程
        for key in self.order_dict.keys():
            if "video_param" == key:
                self.logger.info(">>>>>初始化视频处理模块<<<<<<")
                self.video = build_from_cfg(cfg.video_param, VIDEOPROCESS)
                self.pipline.append(self.__video_processor)
                self.image_folder_path = cfg.video_param.dst_img_path
            if "det_param" == key:
                self.logger.info(">>>>>初始化目标检测模块<<<<<<")
                self.detector = build_from_cfg(cfg.det_param.model, DETECTOR)
                self.pipline.append(self.__det_processor)
                # 如果没有视频处理的流程，则需要从配置文件中定义image_folder_path
                if self.image_folder_path is None:
                    self.image_folder_path = cfg.image_folder_path
            if "cls_param" == key:
                self.logger.info(">>>>>初始化目标分类模块<<<<<<")
                self.classifiers = []
                for model_set in cfg.cls_param.model_set:
                    self.classifiers.append(build_from_cfg(model_set.model, CLASSIFIER))
                self.pipline.append(self.__cls_processor)
                if self.image_folder_path is None:
                    self.image_folder_path = cfg.image_folder_path
            if "crop_param" == key:
                self.logger.info(">>>>>初始化图片截取模块<<<<<<")
                self.img_crop = build_from_cfg(cfg.crop_param, CROPPER)
                self.pipline.append(self.__image_cropper)

    def __image_cropper(self, image_files, rect_rets, image_root_folder):
        self.logger.info("=========开始执行图片截取操作=========")
        crop_image_files = self.img_crop.crop_image(image_files, rect_rets, image_root_folder)
        return crop_image_files

    def __video_processor(self):
        self.logger.info("=========开始执行视频截帧操作=========")
        if os.path.exists(self.image_folder_path):
            self.logger.info(f"截取图片路径已经存在，不再重复截取{self.image_folder_path}")
        else:
            self.video.extract_frames()
        return None

    def __det_processor(self):
        self.logger.info("=========开始执行图像检测=========")
        # 遍历获取文件路径
        image_path_list = get_full_image_path(self.image_folder_path, self.logger)
        self.logger.info(f"{self.image_folder_path} 路径下的图片数量为 {len(image_path_list)}")
        img_det_pred_pkl = f"{self.image_folder_path}_det_pred.pkl"
        if not os.path.exists(img_det_pred_pkl):
            # 推理检测结果
            img_files, det_preds, img_shapes = self.det_inference(self.detector, image_path_list)
            with open(img_det_pred_pkl, "wb") as f:
                pickle.dump([img_files, det_preds, img_shapes], f)
            f.close()
            self.logger.info(f"图片检测完成，中间结果保存在{img_det_pred_pkl}")
        else:
            self.logger.info(f"已存在中间结果{img_det_pred_pkl}，无需再次检测")
            with open(img_det_pred_pkl, "rb") as f:
                img_files, det_preds, img_shapes = pickle.load(f)
            f.close()

        self.logger.info(f"一共{len(img_files)}张图片")
        class_num = self.detector.class_num
        rect_num = np.zeros(class_num)
        # 写入预标注文件
        for i in range(len(img_files)):
            rect_num += self.detector.dump_xml(self.image_folder_path, img_files[i],
                                               img_shapes[i], det_preds[i], class_num)
        self.logger.info(f"一共检测到{rect_num.sum()}个目标")
        for i in range(rect_num.shape[0]):
            self.logger.info(f"第{i}类检测到{rect_num[i]}个目标")

        # 进行Mining
        if "mining_param" in self.order_dict["det_param"].keys():
            self.detector.mining_ret(self.image_folder_path, det_preds, img_files,
                                     self.order_dict["det_param"]["mining_param"])
        return img_files, det_preds, img_shapes

    def __cls_processor(self, crop_image_files=None):
        self.logger.info("=========开始执行分类操作=========")
        img_cls_pred_pkl = f"{self.image_folder_path}_cls_pred.pkl"

        if not os.path.exists(img_cls_pred_pkl):
            if crop_image_files is None:
                crop_image_files = get_full_image_path(self.image_folder_path, self.logger)
            if len(crop_image_files) > 0:
                if not os.path.exists(img_cls_pred_pkl):
                    predict_cls = self.cls_inference(crop_image_files, self.classifiers)
                    with open(img_cls_pred_pkl, "wb") as f:
                        pickle.dump([crop_image_files, predict_cls], f)
                    self.logger.info(f"分类检测完成，中间结果保存在{img_cls_pred_pkl}")
            else:
                self.logger.info("图片数量为0，不进行分类操作")
        else:
            self.logger.info(f"已存在中间结果{img_cls_pred_pkl}，无需再次分类")
            with open(img_cls_pred_pkl, "rb") as f:
                crop_image_files, predict_cls = pickle.load(f)
        mining_param = self.cfg.cls_param.mining_param
        self.classifiers[0].divide_image(self.image_folder_path, crop_image_files, predict_cls, mining_param)
        return None

    def run_pipline(self):
        img_files, det_preds, img_shapes, crop_image_files = None, None, None, None
        for step in self.pipline:
            if step == self.__image_cropper:
                crop_image_files = step(img_files, det_preds, self.image_folder_path)
            elif step == self.__det_processor:
                img_files, det_preds, img_shapes = step()
            elif step == self.__cls_processor:
                ret = step(crop_image_files)
            else:
                ret = step()









