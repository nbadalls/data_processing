"""
单张图像的前向推理结果，用于demo的显示
"""


class Inference:
    def __init__(self, cfg, logger):
        # 根据配置文件完成分类器和检测器的类别初始化
        self.pipline = []

    def __det_processor(self):
        pass

    def __cls_processor(self):
        pass

    def __image_cropper(self):
        pass

    @staticmethod
    def struct_ret(det_preds, cls_preds):
        """
        返回结构体
        [{
        "det": [x,y,w,h],
        "cls": "行人"
        "thre": "0.8"},
        {
        "det": [x,y,w,h],
        "cls": "行人"
        "thre": "0.8"},
        ]
        """
        ret = {}
        return ret

    def run_pipline(self, image):
        cls_preds, det_preds, crop_images = None, None, None
        for step in self.pipline:
            if step == self.__image_cropper:
                crop_images = step(image, det_preds)
            elif step == self.__det_processor:
                 det_preds = step(image)
            elif step == self.__cls_processor:
                cls_preds = step(crop_images)
        ret = self.struct_ret(det_preds, cls_preds)
        return ret


