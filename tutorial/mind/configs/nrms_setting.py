# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: nrms_setting.py
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
import os

# third packages


# my packages


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

print(BASE_DIR)

epochs = 10
seed = 42
batch_size = 32

# Options: demo, small, large
MIND_type = 'large'

# data setting
data_root = "/home/ericdoug_qi/competitions/mind/datas/large"


