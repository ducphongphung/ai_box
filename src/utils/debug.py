import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def get_logger(name, level=None, print_stdout=False):
    logger = logging.getLogger(name)

    try:
        os.makedirs('../../log/')
    except:
        pass

    # if level is not specified, use level of the root logger
    if level is None:
        level = logging.getLogger().getEffectiveLevel()

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    file_h = RotatingFileHandler(
        '../../log/{}_{}.log'.format(logging.getLevelName(level), name),
        mode='a', maxBytes=5*1024*1024, backupCount=2, encoding=None, delay=0)
    file_h.setFormatter(formatter)

    if print_stdout:
        stdout_h = logging.StreamHandler(sys.stdout)
        stdout_h.setFormatter(formatter)
        logger.addHandler(stdout_h)

    logger.addHandler(file_h)
    logger.setLevel(level)

    return logger