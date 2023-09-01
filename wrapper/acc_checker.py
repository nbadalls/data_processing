"""
模型精度校验的pipline
"""
import os, pickle
from pathlib import Path
from module.registry import build_from_cfg
from module.builder import DETECTOR, CLASSIFIER, MODEL_EVAL
from .base_pipline import PipBase


class WrapperEval(PipBase):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.image_folder_path = Path(cfg.log_file).parent
        # 按照配置文件的顺序依次添加执行流程
        for key in self.order_dict.keys():
            if "det_param" == key:
                self.logger.info(">>>>>初始化目标检测模块<<<<<<")
                self.detector = build_from_cfg(cfg.det_param.model, DETECTOR)
                self.pipline.append(self.__det_pred)
                self.image_files = self.__get_det_files()

            if "cls_param" == key:
                self.logger.info(">>>>>初始化目标分类模块<<<<<<")
                self.classifiers = []
                for model_set in cfg.cls_param.model_set:
                    self.classifiers.append(build_from_cfg(model_set.model, CLASSIFIER))
                self.pipline.append(self.__cls_pred)
                self.image_files = self.__get_cls_files()

            if "cls_eval_param" == key:
                self.logger.info(">>>>>初始化分类结果评估<<<<<")
                self.cls_eval = build_from_cfg(cfg.cls_eval_param, MODEL_EVAL)
                self.pipline.append(self.__cls_eval)

            if "det_eval_param" == key:
                self.logger.info(">>>>>初始化检测结果评估<<<<<")
                self.det_eval = build_from_cfg(cfg.det_eval_param, MODEL_EVAL)
                self.pipline.append(self.__det_eval)

    def __get_cls_files(self):
        """
        文件格式：file_name.jpg 0 0 0 1
        """
        file_path = self.cfg.gt_path
        root_path = self.cfg.gt_root_path
        f = open(file_path, "r")
        data = f.read().splitlines()
        f.close()
        image_files = []
        for line in data:
            image_path = f"{root_path}/{line.split(' ')[0]}"
            image_files.append(image_path)
        return image_files

    def __get_det_files(self):
        """
        文件格式：
        # prefix/image_name.jpg
        x1 y1 x2 y2 label x1 y1 x2 y2 label x1 y1 x2 y2 label
        """
        file_path = self.cfg.gt_path
        root_path = self.cfg.gt_root_path
        f = open(file_path, "r")
        data = f.read().splitlines()
        f.close()
        image_files = []
        for line in data:
            if line.find("# ") >= 0:
                image_path = f"{root_path}/{line.split('# ')[1]}"
                image_files.append(image_path)
        return image_files

    def __cls_pred(self):
        pkl_path = f"{self.image_folder_path}/cls.pkl"
        if not os.path.exists(pkl_path):
            predict_cls = self.cls_inference(self.image_files, self.classifiers)
            with open(pkl_path, "wb") as f:
                pickle.dump(predict_cls, f)
            self.logger.info(f"分类检测完成，中间结果保存在{pkl_path}")
        else:
            self.logger.info(f"已存在中间结果{pkl_path}，无需再次分类")
            with open(pkl_path, "rb") as f:
                predict_cls = pickle.load(f)
        return [self.image_files, predict_cls]

    def __det_pred(self):
        det_ret_path = f"{self.image_folder_path}/det_ret.txt"
        if not os.path.exists(det_ret_path):
            img_files, det_preds, img_shapes = self.det_inference(self.detector, self.image_files)
            # 将检测结果写入文件
            f = open(f"{self.image_folder_path}/det_ret.txt", "w")
            for i in range(len(img_files)):
                f.write(f"{img_files[i]}+{int(img_shapes[i][0])}-{int(img_shapes[i][1])} ")
                for elem in det_preds[i]:
                    # x1,y1,x2,y2,conf,cls
                    f.write(f"{elem[0]} {elem[1]} {elem[2]} {elem[3]} {elem[4]} {elem[5]+1} ")
                f.write(f"\n")
            f.close()
        else:
            self.logger.info(f"已存在检测中间结果{det_ret_path}，无需再次检测")
        return det_ret_path

    def __cls_eval(self, predict_cls):
        self.cls_eval.eval(self.cfg.gt_path, predict_cls)

    def __det_eval(self, det_ret_path):
        self.det_eval.eval(self.cfg.gt_path, det_ret_path)

    def run_pipline(self):
        cls_pred_ret, det_ret_path = None, None
        for step in self.pipline:
            if step == self.__cls_pred:
                cls_pred_ret = step()
            elif step == self.__cls_eval:
                 step(cls_pred_ret)
            elif step == self.__det_pred:
                det_ret_path = self.__det_pred()
            elif step == self.__det_eval:
                step(det_ret_path)
            else:
                step()



