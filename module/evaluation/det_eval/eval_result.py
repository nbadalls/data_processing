# -*- coding: utf-8 -*-
# @Time : 2022/2/11 上午9:50
# @Author : zhangkaixiang
# @Company : Minivision
# @File : eval_result.py
# @Software : PyCharm

import numpy as np
from prettytable import PrettyTable
from .match_gt_det import process_batch
from .data_io import read_det_gt_result
from .metrics_mini import ap_per_class_mini


def evaluation(gt_path, det_path, ignore_param, iouv_param, ioa_thre, thre_pr, thre_roc, names, plot, save_dir,
               relax_mode):
    """
    gt_path: 标注文件路径
    det_path： 检测结果路径
    ignore_param： 忽略参数 (30, 1920, 1080) 1920x1080下忽略30x30以下的目标
    iouv_thre： (0.5, 0.95, 10) 表示在gt与检测框匹配阈值在0.5~0.95的范围内，等间隔取10份，取第一个值作为计算静态指标的阈值。
    ioa_thre:  与忽略框匹配的ioa的阈值
    thre_pr:  输出的PR曲線precious的靜態指標 (0.99, 0.88, 0.80, 0.75, 0.70, 0.65)
    thre_roc: 输出的ROC曲線fpr的靜態指標 (0.8, 0.01, 0.02, 0.001)
    names: 類別名稱對應的字典  {1: "bottle", 2: "plastic", 3: "box", 4: "scrap"}
    """
    single_label = True if len(names) == 1 else False
    image_info = read_det_gt_result(det_path, gt_path, ignore_param, single_label=single_label)
    stats = []
    seen = 0

    iouv_thre = np.linspace(*iouv_param)
    for image_name, result in image_info.items():
        det_result = result["det"]
        gt_result = result["gt"]
        match_result, label_match = process_batch(det_result, gt_result, iouv_thre, ioa_thre, relax_mode)
        # match_result, label_match = process_batch(det_result, gt_result, iouv_thre, ioa_thre)
        stats.append((match_result, det_result[:, 4], det_result[:, 5],
                      gt_result[:, 0], label_match[:, 0]))  # (correct, conf, pcls, tcls, tcls_match)
        seen += 1
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    # 去除与忽略框
    i, j = stats[0][:, 0] >= 0, stats[4] >= 0
    tp, conf, pred_cls, target_cls = stats[0][i], stats[1][i], stats[2][i], stats[3][j]
    p, r, ap, f1, ap_class, pr_result, roc_result = ap_per_class_mini(tp, conf, pred_cls, target_cls,
                                                                  thre_pr, thre_roc, len(image_info),
                                                                plot=plot, save_dir=save_dir, names=names)
    # Get map result
    nc = ap.shape[0]
    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(target_cls.astype(np.int64), minlength=nc)  # number of targets per class

    map_tb = PrettyTable()
    map_tb.field_names = ['Class', 'Images', 'Labels', 'P', 'R', f'mAP@{iouv_thre[0]}', f'mAP@{iouv_thre[0]}:{iouv_thre[-1]}']
    map_tb.add_row(['all', seen, nt.sum(), f"{mp:.4f}", f"{mr:.4f}", f"{map50:.4f}", f"{map:.4f}"])
    # Print results per class
    for i, c in enumerate(ap_class):
        map_tb.add_row([names[c], seen, nt[c], f"{p[i]:.4f}", f"{r[i]:.4f}", f"{ap50[i]:.4f}", f"{ap[i]:.4f}"])

    # Get statistic result
    det_n, fp_n, tp_n, det_ig, det_dup_ig, gt_n, fn_n, set_gt_ig = {}, {}, {}, {}, {}, {}, {}, {}
    for c in ap_class:
        det_n[c], fp_n[c], det_ig[c], det_dup_ig[c], gt_n[c], fn_n[c], set_gt_ig[c] = 0, 0, 0, 0, 0, 0, 0

    state_tb = PrettyTable()
    field_names = ["class", "det", "tp", "fp", "det_ig", "fn", "gt", "set_gt_ig"]
    if relax_mode:
        field_names.append("det_dup_ig")
    state_tb.field_names = field_names
    tpfp, pcls, tcls, tcls_m = stats[0][:, 0], stats[2], stats[-2], stats[-1]
    for c in ap_class:
        det_n[c] = len(tpfp[pcls == c])
        tp_n[c] = len(tpfp[(tpfp == 1) & (pcls == c)])
        fp_n[c] = len(tpfp[(tpfp == 0) & (pcls == c)])
        det_ig[c] = len(tpfp[(tpfp == -1) & (pcls == c)])
        fn_n[c] = len(tcls[(tcls_m == 0) & (tcls == c)])
        gt_n[c] = len(tcls[tcls == c])
        set_gt_ig[c] = len(tcls[(tcls == c) & (tcls_m == -2)])
        row_data = [f"{c}-{names[c]}", det_n[c], tp_n[c], fp_n[c], det_ig[c], fn_n[c], gt_n[c], set_gt_ig[c]]
        if relax_mode:
            det_dup_ig[c] = len(tpfp[(tpfp == -2) & (pcls == c)])
            row_data.append(det_dup_ig[c])
        state_tb.add_row(row_data)
    # print(map_tb)
    # print(state_tb)
    return str(map_tb), str(state_tb), pr_result, roc_result, [mp, mr, map50, map]
