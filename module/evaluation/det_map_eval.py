from .det_eval.eval_result import evaluation
from ..logger_manager import get_root_logger
from ..builder import MODEL_EVAL


@MODEL_EVAL.register_module
class DetMapEval:
    def __init__(self, ignore_param, iouv_param, ioa_thre, thre_pr, thre_roc, names, relax_mode):
        self.ignore_param = ignore_param
        self.iouv_param = iouv_param
        self.ioa_thre = ioa_thre
        self.thre_pr = thre_pr
        self.thre_roc = thre_roc
        self.names = {i+1: names[i] for i in range(len(names))}
        self.relax_mode = relax_mode
        self.logger = get_root_logger()

    def eval(self, gt_path, det_path):
        map_tb, state_tb, pr_result, roc_result, _ = evaluation(gt_path, det_path, self.ignore_param, self.iouv_param,
                                                                   self.ioa_thre, self.thre_pr, self.thre_roc,
                                                                   self.names, False, "", self.relax_mode)
        self.logger.info(map_tb)
        self.logger.info(state_tb)
        self.logger.info(pr_result["str"])
        self.logger.info(roc_result["str"])
        return None
