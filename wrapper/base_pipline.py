"""
模型处理数据的pipline
"""

import numpy as np
import torch
from tqdm import tqdm
from module.registry import build_from_cfg
from module.builder import LOADER
from torch.utils.data import DataLoader
from collections import OrderedDict


class PipBase:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.pipline = []
        self.logger = logger
        self.image_folder_path = None
        self.order_dict = OrderedDict(cfg.items())

    def det_inference(self, detector, image_path_list):
        self.cfg.det_param["loader"].image_list = image_path_list
        det_loader = build_from_cfg(self.cfg.det_param.loader, LOADER)
        data_loader = DataLoader(det_loader, pin_memory=True, **self.cfg.det_param.batch_param)
        img_files, det_preds = [], []
        img_shapes = torch.zeros((0, 2))
        for img, img_file, shape in tqdm(data_loader):
            pred = detector.forward(img, shape)
            img_shapes = torch.cat((img_shapes, shape), 0)
            img_files += img_file
            det_preds += pred
        img_shapes = img_shapes.numpy()
        return img_files, det_preds, img_shapes

    def cls_inference(self, crop_image_files, classifiers):
        collect_ret = []
        for i, model_set in enumerate(self.cfg.cls_param.model_set):
            model_set.loader["image_list"] = crop_image_files
            loader = build_from_cfg(model_set.loader, LOADER)
            # 初始化Dataloader
            data_loaders = DataLoader(loader, pin_memory=True, **self.cfg.cls_param.batch_param)
            predict_cls = None
            for img, img_file in tqdm(data_loaders):
                pred_ret = classifiers[i].forward(img)
                predict_cls = pred_ret if predict_cls is None else np.concatenate((predict_cls, pred_ret), axis=0)
            collect_ret.append(predict_cls)
        predict_cls = np.zeros_like(collect_ret[0])
        for elem in collect_ret:
            predict_cls += elem / len(collect_ret)
        return predict_cls











