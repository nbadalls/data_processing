import argparse
import json
import importlib.util
from module.logger_manager import get_root_logger
from wrapper.acc_checker import WrapperEval
from module.utility import make_if_not_exists
from pathlib import Path
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="预处理的配置文件路径")
    args = parser.parse_args()
    # 根据配置文件导入包
    spec = importlib.util.spec_from_file_location('config_module', args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    pipline_config = config_module.config
    make_if_not_exists(Path(pipline_config.log_file).parent)
    logger = get_root_logger(log_file=pipline_config.log_file, file_mode='a')
    logger.info(f"配置文件路径：{config_module.__file__}")
    logger.info(json.dumps(pipline_config, indent=4, ensure_ascii=False))
    wrapper = WrapperEval(pipline_config, logger)
    try:
        wrapper.run_pipline()
    except Exception as e:
        logger.error("An exception occurred: %s", str(e))
        logger.error(traceback.format_exc())
