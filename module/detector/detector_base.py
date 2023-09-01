import sys

import numpy as np

sys.path.append("../../")
from abc import ABC, abstractmethod
from pathlib import Path
import xml.etree.ElementTree as ET
from module.utility import make_if_not_exists, OPERATORS
from xml.dom import minidom  # 导入 minidom 模块
import re
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat


class DetBase(ABC):

    def __init__(self, annotation_dict=None, *args, **kwargs):
        """
        :param annotation_dict:  Meaning of the label {0: "清洗",  1: "未清洗"}
        """
        self.annotation_dict = annotation_dict

    @abstractmethod
    def forward(self, image):
        pass

    def mining_ret(self, root_path, det_result, img_files, mining_cfg):
        """
        root_path: 原始图片的路径
        根据配置文件对满足要求的图片进行Mining
        mining_cfg=edict(
            thre=["0>0.8", "1>0.8", "2>0.8", "3>0.8"],  # 每一类一个Mining的阈值
            ratio=0.8,   # 每一类大于阈值且满足比例的时候才进行Mining
            pool_num=5,
        ),
        """
        label_thre_dict, max_gt = {}, 0
        for elem in mining_cfg.thre:
            parttern = re.match(r'(\d+)(\W+)(\d+\.\d+)', elem)
            gt, operator, thre = parttern.group(1), parttern.group(2), float(parttern.group(3))
            label_thre_dict[gt] = [operator, thre]  # "label": ["<", "0.5"]
            max_gt = max(max_gt, int(gt))

        mining_num = np.zeros(max_gt+1)
        mining_files, empty_files = [], []
        for i in range(len(img_files)):
            mark = np.zeros(max_gt+1)
            if len(det_result[i]) == 0:
                empty_files.append(img_files[i])

            for l in range(max_gt):
                index = (det_result[i][:, 5] - l) == 0  # 筛选标签
                opt, thre = label_thre_dict[str(l)][0], label_thre_dict[str(l)][1]  # 获取每一类的运算符号和阈值
                ret = OPERATORS[opt](det_result[i][index, 4], thre)
                if float(np.sum(ret))/(ret.shape[0]+1e-5) >= mining_cfg.ratio:  # 当某类标签的框满足比例要求的时候，选择Mining的图片
                    mark[int(l)] = 1

            if mark.sum() > 0:
                mining_files.append(img_files[i])
                mining_num += mark  # 统计每一类Mining的图片数量

        self.logger.info(f"从{len(img_files)}张图片中Mining出{len(mining_files)}张样本图片，"
                         f"滤除空图片{len(empty_files)}张，"
                         f"Mining掉{len(img_files)-len(empty_files)-len(mining_files)}张，其中:\n")
        for gt in label_thre_dict.keys():
            self.logger.info(f"第{gt}类，{mining_num[int(gt)]}张")

        # 拷贝Mining出来的图像
        pool_num = mining_cfg.pool_num
        param_str = ""
        for elem in mining_cfg.thre:
            param_str += elem+"-"
        param_str = f"{param_str}r{mining_cfg.ratio}"

        dst_image_folder = f"{root_path}_mining_{param_str}_img"
        dst_annotation_folder = f"{root_path}_mining_{param_str}_annotation"
        dst_empty_image_folder = f"{root_path}_empty_img"
        make_if_not_exists(dst_empty_image_folder)
        make_if_not_exists(dst_image_folder)
        make_if_not_exists(dst_annotation_folder)

        # 拷贝Minig数据
        with Pool(pool_num) as p:
           par = tqdm(p.imap_unordered(self.copy_image, zip(repeat(root_path), mining_files,
                                                            repeat(dst_image_folder),
                                                            repeat(dst_annotation_folder), repeat(True))))
           for _ in par:
               pass

        # 拷贝空图片
        with Pool(pool_num) as p:
           par = tqdm(p.imap_unordered(self.copy_image, zip(repeat(root_path), empty_files,
                                                            repeat(dst_empty_image_folder),
                                                            repeat(None), repeat(False))))
           for _ in par:
               pass
        self.logger.info(f"已完成数据Mining")

    @staticmethod
    def copy_image(arg):
        root_path, mining_file, dst_image_folder, dst_annotation_folder, copy_ann = arg
        img_prefix = mining_file.split(root_path)[-1]
        file_extension = Path(img_prefix).suffix
        img_prefix = img_prefix.replace(file_extension, ".xml")
        dst_image_path = f"{dst_image_folder}/{Path(mining_file).name}"
        shutil.copy(mining_file, dst_image_path)
        if copy_ann:
            # 根据生成标注文件的命名规则，通过原始图片的路径查找对应标注文件的路径
            # image/a/a_xcy.jpg
            # image_annotation/a/a_xcy.xml
            annotation_file = f"{root_path}_annotation{Path(img_prefix).parent}/{Path(img_prefix).name}"
            dst_annotation_path = f"{dst_annotation_folder}/{Path(annotation_file).name}"
            shutil.copy(annotation_file, dst_annotation_path)

    def dump_xml(self, root_path, image_path, image_size, det_result, class_num):
        """
        :param root_path:  root path of source image
        :param image_path: path of source image 
        :param image_size: [h, w]
        :param det_result: [[x1,y1,x2,y2,conf,label], 
                            [x1,y1,x2,y2,conf,label]...]
        """
        root_prefix = f"{root_path}_annotation"
        image_name = Path(image_path).name
        xml_path = f"{Path(image_path.replace(root_path, root_prefix)).parent}/{Path(image_path).stem}.xml"
        make_if_not_exists(Path(xml_path).parent)
        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder')
        folder.text = "folder"
        filename = ET.SubElement(root, 'filename')
        filename.text = image_name
        path = ET.SubElement(root, 'path')
        path.text = "image_path"
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = "Unkonwn"
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_size[1])
        height = ET.SubElement(size, 'height')
        height.text = str(image_size[0])
        depth = ET.SubElement(size, 'depth')
        depth.text = "3"
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = "0"

        rect_sum = np.zeros(class_num)
        for i in range(det_result.shape[0]):
            label =int(det_result[i][5])
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, 'name')

            rect_sum[label] += 1
            if self.annotation_dict is None:
                name.text = str(label)
            else:
                name.text = self.annotation_dict[int(label)]
            pose = ET.SubElement(object, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(det_result[i][0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(det_result[i][1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(det_result[i][2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(det_result[i][3]))

        # 生成格式化的 XML 内容
        xml_str = ET.tostring(root, encoding='utf-8')
        xml_str = xml_str.decode('utf-8')
        dom = minidom.parseString(xml_str)
        formatted_xml = dom.toprettyxml(indent="  ")

        # 将格式化的 XML 内容写入文件
        with open(xml_path, 'w', encoding='utf-8') as xml_file:
            xml_file.write(formatted_xml)
        return rect_sum





