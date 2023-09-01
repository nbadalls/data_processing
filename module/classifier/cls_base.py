import sys
sys.path.append("../../")
from abc import ABC, abstractmethod
import re
import numpy as np
from ..utility import make_if_not_exists, OPERATORS
from pathlib import Path
import shutil
from multiprocessing.pool import Pool
from itertools import repeat
from tqdm import tqdm


class ClsBase(ABC):

    def __init__(self, annotation_dict=None, *args, **kwargs):
        """
        :param annotation_dict:  Meaning of the label {0: "清洗",  1: "未清洗"}
        """
        self.annotation_dict = annotation_dict

    @abstractmethod
    def forward(self, image):
        pass

    def divide_image(self, root_path, image_files, predict_cls, mining_cfg):
        """
        :param root_path: 图片的根路径
        :param image_files: 分类图片路径list
        :param predict_cls: 分类图片的预测结果 np.array() n x cls_num
        :param mining_cfg: Mining配置参数
        :return:
        """
        label_thre_dict, class_name = {}, None
        class_num = predict_cls.shape[1]
        if "annotation_dict" in mining_cfg and mining_cfg.annotation_dict is not None:
            if len(mining_cfg.annotation_dict) == class_num:
                class_name = mining_cfg.annotation_dict
            else:
                self.logger.info(f"类别名称和实际类别的数量不一致，不进行类别替换{len(mining_cfg.annotation_dict)} VS {class_num}")

        dst_path = f"{root_path}_cls_mining"
        for elem in mining_cfg.thre:
            parttern = re.match(r'(\d+)(\W+)(\d+\.\d+)', elem)
            gt, operator, thre = parttern.group(1), parttern.group(2), float(parttern.group(3))
            label_thre_dict[gt] = [operator, thre]  # "label": ["<", "0.5"]

            # 创建分类目标文件夹
            if class_name is None:
                dst_label_path = f"{dst_path}/{gt}{operator}{thre}"
            else:
                dst_label_path = f"{dst_path}/{class_name[int(gt)]}{operator}{thre}"
            make_if_not_exists(dst_label_path)

        # 多类别mining，按照各个类别的阈值进行Mining
        if len(label_thre_dict) > 1:
            pred_labels = np.argmax(predict_cls, axis=1)
            confs = np.max(predict_cls, axis=1)
        elif len(label_thre_dict) == 1:
            gt = int(label_thre_dict.keys()[0])
            confs = predict_cls[:, gt]
            pred_labels = np.ones_like(confs)*gt
        else:
            self.logger.error("没有配置Mining的类别和置信度，异常结束")
            return None

        pool_num = mining_cfg.pool_num
        mining_counter = {str(l): [0, 0] for l in range(class_num)}

        with Pool(pool_num) as p:
            par = tqdm(p.imap_unordered(self.copy_image, zip(repeat(dst_path), image_files,
                                                             confs, pred_labels,
                                                             repeat(label_thre_dict),
                                                             repeat(self.logger), repeat(class_name))))
            for num, label in par:
                mining_counter[label][0] += 1
                mining_counter[label][1] += num
        self.logger.info(f"Mining结果：")
        for key, value in mining_counter.items():
            self.logger.info(f"从第{key}类{value[0]}张图像中Mining出{value[1]}张图像")

    @staticmethod
    def copy_image(args):
        dst_path, image_file, conf, label, label_thre_dict, logger, class_name = args
        opt, thre = label_thre_dict[str(label)][0], label_thre_dict[str(label)][1]
        if class_name is None:
            dst_label_path = f"{dst_path}/{label}{opt}{thre}"
        else:
            dst_label_path = f"{dst_path}/{class_name[int(label)]}{opt}{thre}"

        if OPERATORS[opt](conf, thre):
            dst_image_path = f"{dst_label_path}/{conf:.4f}_{Path(image_file).name}"
            try:
                shutil.copy(image_file, dst_image_path)
            except Exception as e:
                logger.error(f"文件拷贝出现错误：{e}")
            return 1, str(label)
        return 0, str(label)











