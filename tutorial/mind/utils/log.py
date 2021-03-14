# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: log.py
   Description : 
   Author : ericdoug
   date：2021/1/24
-------------------------------------------------
   Change Activity:
         2021/1/24: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import logging

# third packages


# my packages

def get_logger(log_file, log_level=logging.DEBUG):
    """
    定义日志方法
    :param log_file:
    :param log_level:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(log_level)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
