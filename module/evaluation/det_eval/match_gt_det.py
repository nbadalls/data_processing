# -*- coding: utf-8 -*-
# @Time : 2022/1/19 下午4:37
# @Author : zhangkaixiang
# @Company : Minivision
# @File : match_gt_det.py
# @Software : PyCharm
import sys
import numpy as np
"""
1. 检测框与标注的匹配，划分tp fp fn
2. process_batch 增加relax模式：
  适用于比如漂浮物、占到经营类似，由于目标的形状宽泛，正样本定义不严格，
导致满足匹配iou的检测框也可以算作正样本。为放宽评价指标的限制，设置relax为True，会将除最佳匹配之外且
满足iou要求的检测框设置为忽略，不计做误检（fp）。
"""


def box_area(box):
    """
    box: n x 4 numpy
    """
    return (box[:, 2:]-box[:, :2]).prod(1)


def box_iou(box1, box2):
    """
    box1: n x 4 numpy
    box2: m x 4 numpy
    """
    x1y1 = np.maximum(box1[:, None, :2], box2[:, :2])  # n x m x 2
    x2y2 = np.minimum(box1[:, None, 2:], box2[:, 2:])  # n x m x 2
    inter = np.maximum(x2y2-x1y1, 0).prod(2)
    area1 = box_area(box1)
    area2 = box_area(box2)
    return inter / (area1[:, None] + area2[None, :] - inter)


def box_ioa(box1, box2):
    """
    box1: n x 4 numpy
    box2: m x 4 numpy
    """
    x1y1 = np.maximum(box1[:, None, :2], box2[:, :2])  # n x m x 2
    x2y2 = np.minimum(box1[:, None, 2:], box2[:, 2:])  # n x m x 2
    inter = np.maximum(x2y2 - x1y1, 0).prod(2)
    area2 = box_area(box2)
    return inter / area2


def process_batch(detections, labels, iouv_thre, ioa_thre, relax_mode=False):
    """
    得到每张图片的正负样本匹配结果
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 6]), class, x1, y1, x2, y2 set_ig
        iouv_thre (Array[P]) 一组iou值从小到大排列，比如0.5, 0.6, 0.7, 0.8, 0.9，用于计算map
        ioa_thre  float  计算检测框与忽略框的ioa值，超过阈值的认为匹配成功
        relax_mode 如果为True，将满足iou条件的fp设置成忽略，False则正常处理
    Returns:
        det_m, (Array[N, P]), 在不同iou下，检测结果的匹配标识   1-匹配  0-未匹配 -1 忽略
        gt_m, (Array[M, 1]),  标签结果的匹配标识   1-匹配  0-未匹配 -1 忽略 -2 设置ignore_param忽略的正样本

    """
    if not isinstance(iouv_thre, type(np.array([0]))):
        iouv_thre = np.array(iouv_thre)
    gt_m = np.zeros((labels.shape[0], 1), dtype=np.int)
    det_ig_m = np.zeros((detections.shape[0], 1), dtype=np.int)
    det_m = np.zeros((detections.shape[0], iouv_thre.shape[0]), dtype=np.int)

    # 检测框与正样本iou匹配
    iou = box_iou(labels[:, 1:5], detections[:, :4])
    idx = np.where((iou > iouv_thre[0]) & (labels[:, 0:1] == detections[:, 5]))

    # 检测框与忽略框ioa匹配
    ioa = box_ioa(labels[:, 1:5], detections[:, :4])
    idx_g = np.where((ioa > ioa_thre) & (labels[:, 0:1] < 0))

    # 正样本与gt匹配
    if idx[0].shape[0] > 0:
        matches = np.concatenate((np.stack((idx[0], idx[1]), 1), iou[idx[0], idx[1]][:, None]), 1)   # [label, det, iou]
        # 排序为了去重的时候，优先选择iou大的索引
        matches = matches[matches[:, 2].argsort()[::-1]]
        # 去除检测框与多个gt匹配，仅保留一个
        unique_det_idx = np.unique(matches[:, 1], return_index=True)[1]
        matches_uniq = matches[unique_det_idx]

        # 去除gt与多个检测框匹配，仅保留一个
        unique_gt_idx = np.unique(matches_uniq[:, 0], return_index=True)[1]
        matches_uniq = matches_uniq[unique_gt_idx]
        # 标记匹配的正样本
        pos_i = matches_uniq[:, 1].astype(np.int)
        det_m[pos_i] = matches_uniq[:, 2:3] >= iouv_thre
        # 标记gt中匹配上的正样本
        gt_m[matches_uniq[:, 0].astype(np.int)] = 1

        if relax_mode:
            # 将索引映射回matches矩阵
            unique_idx = unique_det_idx[unique_gt_idx]
            # 匹配上tp之外的det结果（包含一个det匹配多个gt的情况）
            rest_det_idx = np.setdiff1d(np.arange(0, len(idx[0])), unique_idx)
            matches_rest = matches[rest_det_idx]
            fp_ig_i = matches_rest[:, 1].astype(np.int)
            # 如果det已经与gt匹配，则不进行忽略（包含一个det匹配多个gt的情况）
            # -2表示被忽略的fp
            det_m = det_m * 3
            det_m[fp_ig_i] += (matches_rest[:, 2:3] >= iouv_thre)*-2
            det_m = det_m.clip(-2, 1)

    # 标记gt中的忽略样本
    gt_m[labels[:, 0] < 0] = -1
    # 从未匹配的正样本中标记set_ig样本(fn 标记为忽略)
    gt_m[(gt_m[:, 0] == 0) & (labels[:, 5] == 1)] = -2

    # 正样本与忽略框匹配
    if idx_g[0].shape[0] > 0:
        ig_matches = np.concatenate((np.stack((idx_g[0], idx_g[1]), 1),
                                    ioa[idx_g[0], idx_g[1]][:, None]), 1)   # [label, det, ioa]
        if idx_g[0].shape[0] > 1:
            # 去除检测框与多个gt匹配，仅保留一个
            ig_matches = ig_matches[np.unique(ig_matches[:, 1], return_index=True)[1]]
        # 标记匹配的忽略样本
        ig_i = ig_matches[:, 1].astype(np.int)
        det_ig_m[ig_i] = 1

        # 将没有匹配上正样本，但是与忽略框匹配的样本置位为-1。不影响完成匹配的正样本
        det_m = 2 * det_m - det_ig_m

    tp = len(det_m[:, 0][det_m[:, 0] > 0])
    fn = len(gt_m[gt_m == 0])
    ig = len(gt_m[gt_m == -2])
    gt = len(labels[:, 0][labels[:, 0] > 0])
    assert tp + fn + ig == gt, f"{tp}+{fn}+{ig}=={gt}"
    return det_m.clip(-2, 1), gt_m



# def eval(gt_path, det_path, ignore_param, iouv=[0.5]):
#     image_info = read_det_gt_result(det_path, gt_path, ignore_param, True)
#     for image_name, result in image_info.items():
#         det_result = result["det"]
#         gt_result = result["gt"]
#         w = result["w"]
#         h = result["h"]
#         match_result, label_match = process_batch(det_result, gt_result, iouv, 0.1)
#         image = np.zeros((w, h, 3), dtype=np.uint8)
#         for i in range(gt_result.shape[0]):
#             x1, y1, x2, y2 = gt_result[i][1], gt_result[i][2], gt_result[i][3], gt_result[i][4]
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
#             if label_match[i] == 0:
#                 image = cv2.putText(image, f"{gt_result[i][0]}-FN", (x1 + 20, y1 + 20), 1, fontScale=2,
#                                     color=(255, 255, 255))
#             else:
#                 image = cv2.putText(image, f"{gt_result[i][0]}", (x1+20, y1+20), 1, fontScale=2, color=(255,255,255))
#         for i in range(det_result.shape[0]):
#             x1, y1, x2, y2 = det_result[i][0], det_result[i][1], det_result[i][2], det_result[i][3]
#             x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#
#             if match_result[i, 0] > 0:
#                 color = (0, 255, 0)
#             elif match_result[i, 0] == 0:
#                 color = (0, 0, 255)
#             else:
#                 color = (0, 255, 255)
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             image = cv2.putText(image, f"{det_result[i][-1]}", (x1-1, y1-1), 1, fontScale=2, color=color)
#
#     cv2.imwrite(f"{Path(gt_path).parent}/{image_name}", image)


# if __name__ == "__main__":
#     root = "/home/minivision/WorkSpace/test"
#     gt_path = f"{root}/gts.txt"
#     det_path = f"{root}/det.txt"
#     ignore_param = (0, 1280, 720)
#     iouv = np.array([0.5])
#     eval(gt_path, det_path, ignore_param, iouv=iouv)



