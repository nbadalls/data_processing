from ..utility import path_clip
from ..logger_manager import get_root_logger
from ..builder import MODEL_EVAL
from sklearn.metrics import roc_curve
import numpy as np


def get_roc_ret(score, label, pos_label=1,
                tolerances=[10 ** -4, 5e-4, 10 ** -3, 2e-3, 4e-3, 6e-3, 10 ** -2, 5e-2, 10 ** -1]):

    fpr, tpr, thresholds = roc_curve(label, score, pos_label=pos_label)
    roc_info = []
    info_dict = {}

    # 10**-6,10**-5, 10**-4, 10**-3, 10**-2, 10**-1
    for tolerance in tolerances:
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))

        #save roc info log
        temp_info = 'fpr\t{}\tacc\t{:.5f}\tthreshold\t{:.6f}'.format(tolerance, max_acc, threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = max_acc
    return roc_info, info_dict


class EvalBase:
    def __init__(self, eval_labels, tolerances=None):
        """
        :param gt_path:
        :param eval_labels: list [0,1,2,3] 需要输出指标的标签
        """
        self.logger = get_root_logger()
        self.eval_labels = eval_labels
        if tolerances is None:
            self.tolerances = [10 ** -4, 5e-4, 10 ** -3, 2e-3, 4e-3, 6e-3, 10 ** -2, 5e-2, 10 ** -1]
        else:
            self.tolerances = tolerances


    @staticmethod
    def to_one_hot_label(label):
        """
        :param label: np.array(), dim=2
        :return:
        """
        if np.max(label) > 1 or label.shape[1] == 1:
            max_cls_num = np.max(label) + 1
            eye_array = np.eye(max_cls_num)
            one_hot = eye_array[label[:, 0], :]
            return one_hot
        return label


@MODEL_EVAL.register_module
class ClsROCEval(EvalBase):
    def __init__(self, eval_labels, tolerances):
        super().__init__(eval_labels, tolerances)

    def eval(self, gt_path, pred_ret):
        """
        :param pred_ret: [crop_image_files, predict_cls], crop_image_files-list n, predict_cls-numpy nxcls_num
        :return:
        """
        self.logger.info(f"ROC 评测测试集路径：{gt_path}")
        gt_prefix, gt_labels = self.label_loader(gt_path, keep_depth=2)
        pred_image_files, pred_cls = pred_ret

        self.logger.info(f"类别数量： {gt_labels.shape[1]}, 图像总数： {gt_labels.shape[0]}")
        try:
            assert len(gt_prefix) == len(pred_image_files) and gt_labels.shape == pred_cls.shape, \
                f"预测结果必须与gt相匹配 gt {len(gt_prefix)} VS pred {len(pred_image_files)}  gt {gt_labels.shape} VS pred {pred_cls.shape}"
        except Exception as e:
            self.logger.error(e)

        pred_dict = {}
        ord_pred_cls = np.zeros((0, pred_cls.shape[1]))
        # 将预测的路径和结果按照key-value的方式排列，用于与gt匹配
        for i in range(len(pred_image_files)):
            prefix = path_clip(pred_image_files[i], keep_depth=2)
            pred_dict[prefix] = pred_cls[i, :]
        for elem in gt_prefix:
            ord_pred_cls = np.concatenate((ord_pred_cls, pred_dict[elem][None, :]), axis=0)

        for label in self.eval_labels:
            roc_info, _ = get_roc_ret(pred_cls[:, label], gt_labels[:, label], tolerances=self.tolerances)
            self.logger.info("==========="*4 + f"第{label}类的测试结果：")
            for elem in roc_info:
                self.logger.info(elem)
        return None

    @staticmethod
    def label_loader(gt_path, keep_depth):
        """
        数据标注的格式1：
            a/b/c1.jpg 0 0 0 1
            a/b/c2.jpg 0 0 1 0
        数据标注的格式2：
            a/b/c1.jpg  0
            a/b/c2.jpg  1
            a/b/c3.jpg  2

        :param gt_path:
        :param keep_depth: 图片路径保留的深度
        :return:
        """
        f = open(gt_path, 'r')
        data = f.read().splitlines()
        f.close()

        prefix, gt_labels = [], None
        for line in data:
            split_ = line.strip().split(' ')
            image_prefix = split_[0]
            label = np.array(list(map(int, split_[1:])))
            prefix.append(path_clip(image_prefix, keep_depth))
            if gt_labels is None:
                gt_labels = label[None, :]
            else:
                gt_labels = np.concatenate((gt_labels, label[None, :]), 0)
        # 如果标签不是one hot格式，则进行转化
        gt_labels = EvalBase.to_one_hot_label(gt_labels)
        return prefix, gt_labels



