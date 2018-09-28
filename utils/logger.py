import os
import logging
import shutil


class Logger:
    def __init__(self, config, log_format='%(asctime)-15s %(message)s'):
        if not os.path.exists(config.exp.base):
            os.mkdir(config.exp.base)
        if not os.path.exists(config.exp.model_path):
            os.mkdir(config.exp.model_path)
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(config.exp.log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logger
        self.config = config

    def info(self, message):
        self.logger.info(message)

    def destroy(self):
        shutil.move(self.config.exp.base, self.config.archive_path)
