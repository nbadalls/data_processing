import os,  cv2
from pathlib import Path
from tqdm import tqdm
from module.utility import generate_random_string, make_if_not_exists, VIDEO_FORMAT
from multiprocessing import Pool
from itertools import repeat
from module.builder import VIDEOPROCESS
from ..logger_manager import get_root_logger


@VIDEOPROCESS.register_module
class VideoProcessBase:
    def __init__(self, video_path, dst_img_path, frequency, pool_num, **kwargs):
        """
        :param video_path:    存放视频根目录
        :param dst_img_path:  截帧图片存放路径
        :param frequency:     间隔多少帧保存一张图片
        :param pool_num:      启动多少个进程进行视频解码
        :param kwargs:        可扩展的其他参数
        """
        self.video_path = video_path
        self.dst_img_path = dst_img_path
        self.logger = get_root_logger()
        self.video_list = []
        self.frequency = frequency
        self.pool_num = pool_num
        self.kwargs = kwargs

    def extract_frames(self):
        # 遍历获得视频路径
        self.logger.info(f"视频文件路径：{self.video_path}")
        for root_path, folder_path, file_path in os.walk(self.video_path):
            for elem in tqdm(file_path):
                if Path(elem).suffix in VIDEO_FORMAT:
                    self.video_list.append(f"{root_path}/{elem}")
                else:
                    self.logger.warning(f"{Path(elem).suffix} 不在 {VIDEO_FORMAT}格式中")
        self.logger.info(f"一共{len(self.video_list)}个视频文件")

        # 多进程读取视频截帧
        self.logger.info(f"开始截取图片,存放根目录{self.dst_img_path}")
        sum_num = 0
        with Pool(self.pool_num) as p:
            par = tqdm(p.imap_unordered(self.extracter, zip(self.video_list, repeat(self.dst_img_path),
                                                  repeat(self.logger), repeat(self.frequency))),
                       total=len(self.video_list))
            for num in par:
                sum_num += num
        self.logger.info(f"处理视频{len(self.video_list)}段，共计{sum_num}张图片")

    @staticmethod
    def extracter(args):
        video_path, dst_folder, logger, frequency = args
        logger.info(f"开始读取:  {Path(video_path).name}")
        randn_str = generate_random_string(3)
        # 替换文件名中的" "和"."为""
        stem = Path(video_path).stem.replace(" ", "").replace(".", "")
        image_prefix = f"{stem}_{randn_str}"
        folder_path = f"{dst_folder}/{image_prefix}"
        video = cv2.VideoCapture(video_path)
        if not video.open(video_path):
            logger.error(f"无法打开文件：\n{video_path}")
            return 0

        make_if_not_exists(folder_path)
        count, index = 1, 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            if count % frequency == 0:
                save_path = f"{folder_path}/{image_prefix}_{index:>04d}.jpg"
                cv2.imwrite(save_path, frame)
                index += 1
            count += 1
        video.release()
        logger.info(f"截取{index:d}张图片")
        return index