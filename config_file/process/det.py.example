from easydict import EasyDict as edict
from pathlib import Path

src_image_path = "/home/minivision/DataSet/Image/20230804_抽烟检测_2023_20230829/video_crop_image"
dst_root_path = Path(src_image_path).parent
device = 0
config = edict(
    log_file=f"{dst_root_path}/info_2.log",
    image_folder_path=src_image_path,
    det_param=edict(
        model=edict(
            type="YoloDet",
            model_path="/home/minivision/Model/CNN/20230203055406_Pedestrian_MS288-896_2023-0113-Pedestrian_ig0.5_iouv0.5-0.95_yolov5m_ep59-best_map.onnx",
            thre=[0.5],
            nms_iou=0.3,
            device_id=device,
            annotation_dict=["行人"]),
        loader=edict(
            type="LoadImagesYolo",
            preprocess=edict(
                type="YoloProcess",
                img_size=640,
                rgb=False
            ),
        ),
        batch_param=edict(
            batch_size=64,
            num_workers=10
        ),)
    )