# coding=utf-8
from easydict import EasyDict as edict

dst_root_path = "/home/minivision/DataSet/Eval/Predestrain3"
gt_root_path = "/home/minivision/mount_directory/GPU_CLUSTER/h3c_new/project_dataset"
gt_path = "/home/minivision/mount_directory/GPU_CLUSTER/h3c_new/project_dataset/PedestrianData/Label/VideoStructure/行人汇总/20230217/20220419_行人测试集汇总.txt"
device_id = 0
config = edict(
    log_file=f"{dst_root_path}/eval_info.log",
    gt_root_path=gt_root_path,
    gt_path=gt_path,
    det_param=edict(
        model=edict(
            type="YoloDet",
            # model_path="/home/minivision/Model/CNN/20230203055406_Pedestrian_MS288-896_2023-0113-Pedestrian_ig0.5_iouv0.5-0.95_yolov5m_ep59-best_map.onnx",
            model_path="/home/minivision/Model/CNN/20230802224759_Pedestrian_MS320-672_2023-0802_Pedestrian_easy_ig0.5-sz30_iouv0.5-0.95_yolov5m_ep9-best_map_decode.onnx",
            thre=[0.1],
            device_id=device_id,
            nms_iou=0.3,
            annotation_dict=["行人"]
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
            batch_size=128,
            num_workers=10
        )),
    det_eval_param=edict(
        type="DetMapEval",
        ignore_param=(40, 1920, 1080),
        iouv_param=(0.5, 0.95, 5),
        ioa_thre=0.5,
        thre_pr=(0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.80, 0.75, 0.70, 0.65),
        thre_roc=(0.3, 0.2, 0.1, 0.03, 0.02, 0.01, 0.001),
        names=["行人"],
        relax_mode=False
    )
)