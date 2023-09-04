from easydict import EasyDict as edict
from pathlib import Path

src_image_path = "/home/minivision/DataSet/Image/垃圾检测/video_crop_image_mining_0>0.8-1>0.8-2>0.8-3>0.8-r0.8_img"
dst_root_path = Path(src_image_path).parent
device_id = 0
config = edict(
    log_file=f"{dst_root_path}/info1.log",
    image_folder_path=src_image_path,
    det_param=edict(
        model=edict(
            type="YoloDet",
            model_path="/home/minivision/mount_directory/GPU_CLUSTER/model/Trash/select_model_yolo/20220218144154/Trash_640_2021-08-21-Trash-Aug09-15-SY-c6Neg_ig0.3-sz15_iouv0.4-0.95_masic1.0_s0.5_iou0.3N_SGDR_yolov5s_Feb16-16-21-09/20220216202849_Trash_640_2021-08-21-Trash-Aug09-15-SY-c6Neg_ig0.3-sz15_iouv0.4-0.95_masic1.0_s0.5_iou0.3N_SGDR_yolov5s_ep28-best_map.onnx",
            nms_iou=0.3,
            device_id=device_id,
            thre=[0.5, 0.5, 0.5, 0.5],  # 对应多个类别的阈值，选取大于阈值的结果保存
            annotation_dict=["bottle", "plastic", "box", "scrap"],
        ),
        mining_param=edict(
            thre=["0>0.8", "1>0.8", "2>0.8", "3>0.8"],
            ratio=0.8,
            pool_num=5,
        ),
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