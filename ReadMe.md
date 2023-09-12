# 项目说明
  本项目旨在统一在数据处理环节的流程，包括对视频帧的截取、目标的检测、ROI区域图片截取、目标分类的全环节打通。
  利用注册器的机制，对于每个子功能进行模块化管理，便于后期的功能的添加和扩展。通过配置文件整个各个组件，串联数据处理的流程。
  对于模型的推理统一.onnx格式，通过eval.py验证模型的精度。确保转换模型精度无误时，调用main.py进行数据处理。

# 目录结构
<pre>
  ├── config_file/         配置文件    
  │ ├── eval/              精度核验  
  │ └── process/           数据处理  
  ├── module/              数据处理的子模块  
  │ ├── classifier/        分类器模型加载、推理、数据挖掘
  │ ├── dataset/           DataLoader中的batch图像的处理，以检测和分类区分
  │ ├── detector/          检测器模型加载、推理、数据挖掘
  │ ├── evaluation/        检测和分类模型的评价指标评测
  │ ├── image/             对于图片的操作，截取图片的ROI区域
  │ ├── preprocess/        检测器和分类器推理之前图片的前处理操作
  │ ├── video/             对于视频的操作，截取视频帧
  │ ├── builder.py         对于不同的模块进行注册
  │ ├── logger_manager.py  日志的管理
  │ ├── registry.py        注册器定义
  │ └── utility.py         通用函数的定义
  ├── models/              保存模型的路径，仅支持onnx格式的模型进行推理
  ├── wrapper/
  │ ├── acc_checker.py     串联模型精度校验的流程
  │ ├── base_pipline.py    pipline的基类，定义了检测和分类公用的推理部分
  │ └── processor.py       串联数据处理的流程
  ├── eval.py              执行模型精度校验，匹配 config/eval中的配置文件
  └── main.py              执行数据处理，匹配 config/process中的配置文件
</pre>

# 配置文件说明
根据配置文件的顺序依次执行需要的操作  
#### 目标检测配置文件
```python
from easydict import EasyDict as edict    
det_param=edict(
        model=edict(
            type="YoloDet",                                          # 检测器类名，初始化相应的检测器类
            model_path="model_path.onnx",
            nms_iou=0.3,                                             # 目标检测nms阶段的iou
            device_id=0,
            thre=[0.5, 0.5, 0.5, 0.5],                               # 对应多个类别的阈值，选取大于阈值的结果保存；如果阈值为单个float类型，则自动扩展到所有类别
            annotation_dict=["bottle", "plastic", "box", "scrap"],   # 对应多个类别对应的名称，对应预标注类名
        ),
        mining_param=edict(                                          # 若不配置此参数则不进行Mining      
            thre=["0>0.8", "1>0.8", "2>0.8", "3>0.8"],               # 设置每个类别的Mining条件，注意model.thre的检测参数为默认大于阈值的结果
            ratio=0.8,                                               # 每张图片里某个类别满足条件的比例大于等于ratio的情况下筛选图片
            pool_num=5,                                              # 拷贝Mining图片的进程数
        ),
        loader=edict(
            type="LoadImagesYolo",                                    # 检测图片处理类名
            preprocess=edict(          
                type="YoloProcess",                                   # 数据预处理方法，在module/preprocess/det_yolo_input.py中定义
                img_size=640,                                         # 检测图片的尺寸
                rgb=False                                             # True=rgb， False=bgr
            ),
        ),
        batch_param=edict(
            batch_size=64,                                            # DataLoader的batch_size
            num_workers=10                                            # 加载数据的线程数
        ),
      )
```

#### 目标分类配置文件
```python
from easydict import EasyDict as edict  
cls_param=edict(
    model_set=[
        # Model1=======================================================
        edict(
            loader=edict(
                type="LoadImageCls",                                  # 分类图片处理类名
                preprocess=edict(
                    type="SmokePhoneCrop",                            # 数据预处理方法，在module/preprocess/cls_input.py中扩展定义
                    width=224,
                    height=224,
                    rgb=True)),                                       # True=rgb， False=bgr
            model=edict(
                type="Classifier",                                    # 分类器类名，初始化相应的分类器类
                model_path="model_path1.onnx",
                post_process="sigmoid",                               # 后处理分类 "sigmoid", "softmax", 默认None不需要后处理操作
                device_id=0)
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
                model_path="model_path2.onnx",
                post_process="sigmoid",
                device_id=0))],

    batch_param=edict(
        batch_size=64,                                              # DataLoader的batch_size
        num_workers=10),                                            # 加载数据的线程数
    mining_param=edict(
        thre=["0>0.5", "1>0.5", "2>0.5", "3>0.5", "4>0.5", "5>0.5", "6>0.5"],  # 根据类别Minig不同阈值下图片
        annotation_dict=["行人", "明显打电话", "明显玩手机", "明显抽烟", "不明显打电话", "不明显玩手机", "不明显抽烟"], # 每类代表的标签，若为None，默认使用类别的数字代替
        pool_num=5,  # 拷贝Mining出来图片的进程数
    )
)
```

#### 视频截帧操作配置文件
从视频中截取图片，再进行后续的检测操作
```python
from easydict import EasyDict as edict  
video_param=edict(
    type="VideoProcessBase",                                         # 视频处理类
    video_path="src_video_path",
    dst_img_path=f"dst_root_path/video_crop_image",
    frequency=10,                                                    # 每间隔多少帧截取一张图片
    pool_num=10                                                      # 处理视频的进程数
),
```


#### ROI图片截取配置文件
根据检测框的结果提取ROI图片，用于后续的分类操作
```python
from easydict import EasyDict as edict  
crop_param=edict(
    type="CropImage",                                               # ROI提取类               
    pool_num=10,                                                    # 截取进程数
    scale_w=1.0,                                                    # 扩边参数，宽变为原来的scale_w倍，
    scale_h=1.0)                                                  # 扩边参数，宽变为原来的scale_h倍，
```

# 调用脚本
#### 模型精度核验
在配置文件中配置测试集路径和输出路径，在日志中显示模型在测试集中精度
```shell
# 分类模型精度验证
python eval.py --config config/eval/cls_roc_eval.py

# 检测模型精度验证
python eval.py --config config/eval/det_map_eval.py
```
#### 数据处理
```shell
# 视频截取图片-> 目标检测预标 -> roi图片截取 -> 分类器结果Mining
# 配置文件中各个模块的顺序对应着各个子流程的顺序
python main.py --config config/process/video_det_cls.py
```
输出结果目录结构
<pre>
  ├── ImageResult/                                 输出结果文件夹    
  │  ├── info.log                                  输出文件日志
  │  ├── video_crop_image/                         从视频中截取的图片 
  │  ├── video_crop_image_annotation/              从视频中截取的图片标注
  │  ├── video_crop_image_cls_mining/              截取ROI图片过分类模型Mining的图片结果
  │  ├── video_crop_image_cls_pred.pkl             截取ROI图片过分类的中间结果，如果存在则直接读取，不再重复检测
  │  ├── video_crop_image_crop_sw_1.0_sh_1.0       根据图片标注截取ROI图片
  │  └── video_crop_image_det_pred.pkl             从视频中截取的图片标注的中间结果，如果存在则直接读取，不再重复检测
</pre>


# 注意
1. onnxruntime-gpu需要同时安装对应的cuda和cudnn才能生效，详见：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements



