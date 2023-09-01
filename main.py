from module.logger_manager import get_root_logger
from wrapper.processor import Processor
from module.utility import make_if_not_exists
from pathlib import Path
import json
import argparse
import importlib.util


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
    logger.info(f"配置文件路径：{args.config}")
    logger.info(json.dumps(pipline_config, indent=4, ensure_ascii=False))
    wrapper = Processor(pipline_config, logger)
    wrapper.run_pipline()


