#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name log
@Description
    
@Author LiHao
@Date 2020/1/2
"""
import logging


def get_logger(module_name, level=None):
    logger = logging.getLogger(module_name)
    # fh = logging.FileHandler(module_name+'.log')
    ch = logging.StreamHandler()
    if level is None:
        logger.setLevel(logging.INFO)
        # fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    else:
        logger.setLevel(level)
        # fh.setLevel(level)
        ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
    ch.setFormatter(formatter)
    # fh.setFormatter(fh)
    logger.addHandler(ch)
    # logger.addHandler(fh)
    return logger
