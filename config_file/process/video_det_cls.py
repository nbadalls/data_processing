# coding=utf-8
from easydict import EasyDict as edict

src_video_path = "/home/minivision/DataSet/Video/20230804_抽烟检测"
dst_root_path = "/home/minivision/DataSet/Image/20230804_抽烟检测_T"
device_id = 0
config = edict(
    log_file=f"{dst_root_path}/info.log",
    video_param=edict(
        type="VideoProcessBase",
        video_path=src_video_path,
        dst_img_path=f"{dst_root_path}/video_crop_image",
        frequency=10,
        pool_num=10
    ),
    det_param=edict(
        model=edict(
            type="YoloDet",
            model_path="/home/minivision/Model/CNN/20230203055406_Pedestrian_MS288-896_2023-0113-Pedestrian_ig0.5_iouv0.5-0.95_yolov5m_ep59-best_map.onnx",
            device_id=device_id,
            thre=[0.5],  # 对应多个类别的阈值，选取大于阈值的结果保存
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
            batch_size=64,
            num_workers=10
        ),
    ),

    crop_param=edict(
        type="CropImage",
        pool_num=10,
        scale_w=1.0,
        scale_h=1.0),

    cls_param=edict(
        model_set=[
            # Model1=======================================================
            edict(
                loader=edict(
                    type="LoadImageCls",
                    preprocess=edict(
                        type="SmokePhoneCrop",
                        width=224,
                        height=224,
                        rgb=True)),
                model=edict(
                    type="Classifier",
                    post_process="sigmoid",
                    model_path="/home/minivision/Model/CNN/smoke/2023-07-04-02-20_SmokePhone-PyAMP-MultiLabel-DA_224x224_base_smoke-phone-train-20230703-org_smoke-phone-test-20230703-org_MobileNetV2-c7_model_iter-38554_smoke-0.6645_obsmoke-0.2435_224-224.onnx",
                    device_id=device_id)
            ),

            # Model2=======================================================
            edict(
                loader=edict(
                    type="LoadImageCls",
                    preprocess=edict(
                        type="SmokePhoneCrop",
                        width=384,
                        height=384,
                        rgb=True)),
                model=edict(
                    type="Classifier",
                    post_process="sigmoid",
                    model_path="/home/minivision/Model/CNN/smoke/2023-07-04-03-39_SmokePhone-PyAMP-MultiLabel-DA_384x384_base_smoke-phone-train-20230703-org_smoke-phone-test-20230703-org_MobileNetV2-c7_model_iter-52100_smoke-0.6645_obsmoke-0.2472_384-384.onnx",
                    device_id=device_id))],

        batch_param=edict(
            batch_size=64,
            num_workers=10),
        mining_param=edict(
            thre=["0>0.5", "1>0.5", "2>0.5", "3>0.5", "4>0.5", "5>0.5", "6>0.5"],
            annotation_dict=["行人", "明显打电话", "明显玩手机", "明显抽烟", "不明显打电话", "不明显玩手机", "不明显抽烟"],
            pool_num=5,
        )
    )
)