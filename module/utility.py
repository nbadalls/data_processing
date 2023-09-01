import random
import string
import os
from pathlib import Path
import operator

IMG_FORMAT = [".jpg", ".JPE", ".png", ".PNG", ".jpeg", ".JPEG"]
VIDEO_FORMAT = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"]


def generate_random_string(length):
    # 定义包含所有可能字符的字符串
    all_chars = string.ascii_letters

    # 使用 random.choice() 方法从 all_chars 中随机选择字符，并重复该过程生成所需长度的字符串
    random_string = ''.join(random.choice(all_chars) for _ in range(length))
    return random_string


def make_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_full_image_path(folder_path, logger):
    image_path_list = []
    for root_path, _, file_names in os.walk(folder_path):
        for image_name in file_names:
            if Path(image_name).suffix in IMG_FORMAT:
                image_path_list.append(f"{root_path}/{image_name}")
            else:
                logger.warning(f"{image_name} 的格式不包含在 {IMG_FORMAT}")
    return image_path_list


def get_path_depth(src_path):
    parts = src_path.split("/")
    return len(parts)


def path_clip(src_path, keep_depth):
    """
    保留多深的路径路径名称：
    :param src_path:   a/b/c/d.jpg
    :param keep_depth:  2
    :return: c/d.jpg
    """
    parts = src_path.split("/")
    depth = get_path_depth(src_path)
    keep_depth = min(depth, keep_depth)
    return os.path.join(*parts[-keep_depth:])


OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le
}