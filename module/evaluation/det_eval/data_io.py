# -*- coding: utf-8 -*-
# @Time : 2022/1/6 上午10:51
# @Author : zhangkaixiang
# @Company : Minivision
# @File : data_io.py
# @Software : PyCharm

import numpy as np
from ...utility import path_clip

def read_det_gt_result(det_path, gt_path, ignore_param, xywh=False, single_label=False):
    """
    分别读取检测结果的gt和det结果，按照图片名称保存到统一的结构体中。
    兼容的文件格式如下：
    detection result:
    prefix/image_name.jpg+h-w x1 y1 x2 y2 conf label x1 y1 x2 y2 conf label

    ground truth:
    # prefix/image_name.jpg
    x1 y1 x2 y2 label x1 y1 x2 y2 label x1 y1 x2 y2 label

    注：1. label 统一从1开始
       2. 小于零的标签表示忽略类
       3. set_ig 通过设置ignore_param参数，忽略的样本标记为1，否则为0

    return:
          'prefix/a.jpg':{
                      {
                'w': 1280,
                'h': 720,
                'det': <np.ndarray> (n, 6) in (x1, y1, x2, y2, conf, class) order.
                'gt': <np.ndarray> (n, 6), in (class, x1, y1, x2, y2, set_ig) order.
            },
          }    ...
    ignore_param: 忽略框尺寸大小 (30, 1920, 1080)表示忽略1920x1080下30x30样本
    """

    ig_ratio = float(ignore_param[0]**2) / (ignore_param[1]*ignore_param[2])
    image_info = {}

    # 保存检测结果
    with open(det_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            name = path_clip(line.split("+")[0], 2)
            sp = line.split("+")[-1].split(" ")
            image_size = sp[0].split("-")

            h = int(image_size[0]) if len(image_size) > 0 else 1080
            w = int(image_size[1]) if len(image_size) > 0 else 1920
            bbox = []
            for i in range(len(sp) // 6):
                b = i * 6
                x1, y1, x2, y2, score, label = sp[b+1], sp[b+2], sp[b+3], sp[b+4], sp[b+5], sp[b+6]
                x1, y1, x2, y2, score, label, = map(float, [x1, y1, x2, y2, score, label])
                if xywh:
                    x2, y2 = x1+x2-1, y1+y2-1
                ratio = (y2-y1)*(x2-x1) / (h*w)
                # 舍弃满足忽略条件的检测框
                if ratio > ig_ratio:
                    bbox.append([x1, y1, x2, y2, score, label])

            image_info[name] = {}
            image_info[name]['h'] = h
            image_info[name]['w'] = w
            image_info[name]['det'] = np.array(bbox) if len(bbox) > 0 else np.empty((0, 6))

    # 保存ground truth结果
    gt_num = 0
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "# " not in line:
                continue

            name = path_clip(line[2:].split('\n')[0], 2)  # "# image_path"
            line = lines[min(idx + 1, len(lines)-1)]  # 限制最后一行没有标注的情况
            sp = line.split(' ')
            bbox = []
            h, w = image_info[name]["h"], image_info[name]["w"]
            for i in range(len(sp) // 5):
                b = i * 5
                x1, y1, x2, y2, label = sp[b], sp[b+1], sp[b+2], sp[b+3], sp[b+4]
                x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2, label])
                if xywh:
                    x2, y2 = x1+x2-1, y1+y2-1
                ratio = (y2-y1)*(x2-x1)/(h*w)
                # 通过设置ignore_param参数，忽略的样本标记为1，否则为0
                set_ig = 1 if ratio <= ig_ratio else 0
                label = 1 if single_label and label > 0 else label
                bbox.append([label, x1, y1, x2, y2, set_ig])
            image_info[name]['gt'] = np.array(bbox) if len(bbox) > 0 else np.empty((0, 6))
            gt_num += 1

    assert gt_num == len(image_info), \
        f"det image num {len(image_info)} VS gt image num {gt_num}, UNMATCHED!!"
    return image_info


