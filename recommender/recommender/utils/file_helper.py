# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: file_helper.py
   Description : 
   Author : ericdoug
   date：2021/1/30
-------------------------------------------------
   Change Activity:
         2021/1/30: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import tensorflow as tf
from os import path, listdir

# third packages


# my packages

def decode_csv(line):
    # 按照,分割，取label和feature
    columns = tf.string_split([line], ' ')
    center_words = tf.reshape(tf.string_to_number(columns.values[0], out_type=tf.int32),[-1])
    target_words = tf.reshape(tf.string_to_number(columns.values[1], out_type=tf.int32),[-1])
    return {'center_words': center_words, 'target_words': target_words}


def read_my_file_format(filenames, batch_size, num_epochs=1):
    """文件读取，采用dataset格式

    :param filenames:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    # 读取文件
    dataset = tf.data.TextLineDataset(filenames).map(lambda x: decode_csv(x)).prefetch(batch_size).cache()
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def get_file_list(my_path):
    """文件列表

    :param my_path:
    :return:
    """
    files = []
    if path.isdir(my_path):
        [files.append(path.join(my_path, p)) for p in listdir(my_path) if path.isfile(path.join(my_path, p))]
    else:
        files.append(my_path)
    return files
