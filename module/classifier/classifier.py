import numpy as np

from .cls_base import ClsBase
from module.logger_manager import get_root_logger
import torch
import onnxruntime
from ..builder import CLASSIFIER


@CLASSIFIER.register_module
class Classifier(ClsBase):
    def __init__(self, model_path, device_id, post_process=None):
        self.logger = get_root_logger()
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=[
                                                        ('CUDAExecutionProvider',
                                                         {'device_id': device_id})])
        self.logger.info(f"Classifier init finished\n Model: {model_path}")
        self.post_process = post_process
        try:
            assert post_process in ['softmax', 'sigmoid', None], f"{post_process}的方法不在 softmax sigmoid中"
        except Exception as e:
            self.logger.error(e)

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        pred = self.session.run([self.session.get_outputs()[0].name],
                                {self.session.get_inputs()[0].name: img})[0]
        if self.post_process is not None:
            pred = getattr(self, self.post_process)(pred)
        return pred

    def softmax(self, pred_ret):
        exp_pred = np.exp(pred_ret)
        sum_exp_pred = np.sum(exp_pred, axis=1)
        return exp_pred / sum_exp_pred[:, None]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


