# -*- coding: utf-8 -*-
# @Time : 2022/5/7 下午4:55
# @Author : zhangkaixiang
# @Company : Minivision
# @File : yolo_det.py
# @Software : PyCharm
"""
注意onnxruntime-gpu需要同时安装对应的cuda和cudnn才能生效，详见：
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
"""
import sys
sys.path.append("../../")
import torch
import numpy as np
import onnxruntime
from .detector_base import DetBase
from .yolo_utility import non_max_suppression, scale_coords
from module.logger_manager import get_root_logger
from module.builder import DETECTOR


@DETECTOR.register_module
class YoloDet(DetBase):
    def __init__(self, model_path, thre, device_id=0, nms_iou=0.3, annotation_dict=None):
        self.logger = get_root_logger()
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=[
                                                        ('CUDAExecutionProvider',
                                                         {'device_id': device_id})])
        self.thre = thre
        self.annotation_dict = annotation_dict
        self.logger.info(f"Yolo Detector init finished\n Model: {model_path} \n Threshold: {thre}")
        self.nms_iou = nms_iou
        self.class_num = self.get_class_num()

        # 如果self.thre不是list而是单个的数值，则进行扩展复制
        if isinstance(self.thre, list):
            self.thre = np.array(self.thre)
        elif isinstance(self.thre, float):
            self.logger.info(f"将阈值{self.thre}扩展到{self.class_num}个类别")
            self.thre = np.ones(self.class_num) * self.thre
        else:
            assert type(self.thre) in [list, float], "thre的类型必须是list或者float"
        assert len(self.thre) == self.class_num, f"阈值的个数与类别数必须相等{len(self.thre)} vs {self.class_num}"

    def get_class_num(self):
        img = np.random.randn(1, 3, 640, 640).astype("float32")
        pred = self.session.run([self.session.get_outputs()[0].name],
                                {self.session.get_inputs()[0].name: img})
        pred = torch.tensor(pred[0])  # bxnx(nc+5)
        cls_num = pred.shape[2] - 5
        self.logger.info(f"检测目标的类别数为{cls_num}")
        return cls_num

    def forward(self, img, img0_shape):
        """
        Args:
            img:  BxCxHxW
            img0_shape: BxHxW origin image size
        Returns: numpy  nx6 [x1,y1,x2,y2,conf,cls]
        """
        if isinstance(img, torch.Tensor):
            img = img.numpy().astype("float32")
        pred = self.session.run([self.session.get_outputs()[0].name],
                                {self.session.get_inputs()[0].name: img})
        pred = torch.tensor(pred[0])   # bxnx(nc+5)
        pred = non_max_suppression(pred, 0.1, self.nms_iou, None, False, 1000)

        for i, elem in enumerate(pred):
            # 转化回原图
            elem[:, :4] = scale_coords(img.shape[2:], elem[:, :4], img0_shape[i]).round()
            elem = elem.numpy()
            # 根据每类的阈值筛选满足条件的检测框
            cls_id = elem[:, 5].astype("int")
            conf = elem[:, 4]
            batch_thres = self.thre[cls_id]
            mask = (conf > batch_thres)
            pred[i] = elem[mask, :]
        return pred

    def debug(self, image_path, pred):

        import cv2
        img = cv2.imread(image_path)
        for row in pred:

            cv2.rectangle(img, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0,0,255), 3)
        cv2.imshow("1", img)
        cv2.waitKey(0)


