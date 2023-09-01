# -*- coding: utf-8 -*-
# @Time : 2022/1/24 下午2:54
# @Author : zhangkaixiang
# @Company : Minivision
# @File : metrics_mini.py
# @Software : PyCharm

import os
import sys
sys.path.append("../../../")
import numpy as np
from pathlib import Path
from .metrics import plot_pr_curve, plot_mc_curve, compute_ap
import matplotlib.pyplot as plt
"""
得到tp fp fn，计算相应的pr曲线指标和roc曲线指标
"""


def get_pr_info(r, p, conf, label, thre):
    """
    计算对应precious下的recall和threshold值，保存为字符串输出
    """
    p = np.flip(np.maximum.accumulate(np.flip(p[None, :])))
    abs_val = np.abs(np.array(thre)[:, None] - p)     # mx1 1x1000  -> mx1000
    idxes = np.argmin(abs_val, axis=1)
    pr_info = ""
    pr_info += f"{label} =====================================\n"
    pr_dict = {}
    for i, idx in enumerate(idxes):
        pr_info += f"r {r[idx]:.4f} p {thre[i]:.4f} t {conf[idx]:.4f}\n"
        pr_dict[thre[i]] = r[idx]
    return  pr_info, pr_dict


def get_roc_info(fpr, tpr, conf, label, thre):
    """
    计算对应fpr下的tpr和threshold值，保存为字符串输出
    """
    abs_val = np.abs(np.array(thre)[:, None] - fpr)     # mx1 1x1000  -> mx1000
    idxes = np.argmin(abs_val, axis=1)
    roc_info = ""
    roc_info += f"{label} =====================================\n"
    roc_dict = {}
    for i, idx in enumerate(idxes):
        roc_info += f"fpr {thre[i]:.4f} tpr {tpr[idx]:.4f} t {conf[idx]:.4f}\n"
        roc_dict[thre[i]] = tpr[idx]
    return roc_info, roc_dict


def plot_roc_curve(px, py, save_dir='roc_curve.png', names=()):
    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(fpr, tpr)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_xlim(0, np.max(px))
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def ap_per_class_mini(tp, conf, pred_cls, target_cls, thre_pr, thre_roc,
                      image_num, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        thre_pr:  certain precious to get corresponding recall. Like (0.99, 0.88, 0.80, 0.75, 0.70, 0.65)
        thre_roc: certain fpr to get corresponding tpr. Like (0.8, 0.01, 0.02, 0.001)
        image_num: images in dataset
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    roc_fpr, roc_tpr = np.zeros((nc, 1000)), np.zeros((nc, 1000))
    roc_y = []
    pr_str, roc_str, pr_dict, roc_dict = "", "", [], []
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # FPR = FP / image_num
            fpr = fpc / image_num
            roc_fpr[ci] = np.interp(-px, -conf[i], fpr[:, 0], left=0)
            roc_tpr[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
            if plot:
                roc_y.append(np.interp(px, fpr[:, 0], recall[:, 0]))

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))

            pr_str_c,  pr_dict_c = get_pr_info(r[ci], p[ci], px, names[ci], thre_pr)
            roc_str_c, roc_dict_c = get_roc_info(roc_fpr[ci], roc_tpr[ci], px, names[ci], thre_roc)
            pr_str += pr_str_c
            roc_str += roc_str_c
            pr_dict.append(pr_dict_c)
            roc_dict.append(roc_dict_c)

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            plot_roc_curve(px, roc_y, Path(save_dir) / 'ROC_curve.png', names)
            plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
            plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
            plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
            plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')
        except Exception as e:
            print("Can't plot metric curve, because ", e)

    i = f1.mean(0).argmax()  # max F1 index
    pr_result = {"str": pr_str, "dict": pr_dict}
    roc_result = {"str": roc_str, "dict": roc_dict}
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32'), pr_result, roc_result
