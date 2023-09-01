from .registry import Registry

VIDEOPROCESS = Registry("VideoProcess")
DETECTOR = Registry("Detector")
LOADER = Registry("Loader")
CROPPER = Registry("ImageCrop")

CLASSIFIER = Registry("Classifier")
CLASSIFIER_PREPROCESS = Registry("Classifier_Preprocess")
DETECTION_PREPROCESS = Registry("Detection_Preprocess")

MODEL_EVAL = Registry("ModelEval")
