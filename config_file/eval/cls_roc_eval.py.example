# coding=utf-8
from easydict import EasyDict as edict

# 检测结果存放路径
dst_root_path = "/home/minivision/DataSet/Eval/Smoke2"
# 测试集目录前缀
gt_root_path = ""
# 测试集标签存放路径
gt_path = "/home/minivision/DataSet/TestSet/smoke/TEST_20201211-smoke_7classes.txt"
device = 0
config = edict(
    log_file=f"{dst_root_path}/eval_info.log",
    gt_root_path=gt_root_path,
    gt_path=gt_path,

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
                    device_id=device)
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
                    device_id=device))],

        batch_param=edict(
            batch_size=128,
            num_workers=10),
    ),
    cls_eval_param=edict(
        type="ClsROCEval",
        eval_labels=[0, 1, 2, 3, 4, 5, 6],
        tolerances=None,
    )
)