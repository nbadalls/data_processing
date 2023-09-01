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
  ├── wrapper/  
  │ ├── acc_checker.py     串联模型精度校验的流程
  │ ├── base_pipline.py    pipline的基类，定义了检测和分类公用的推理部分
  │ └── processor.py       串联数据处理的流程
  ├── eval.py              执行模型精度校验，匹配 config/eval中的配置文件
  └── main.py              执行数据处理，匹配 config/process中的配置文件
</pre>