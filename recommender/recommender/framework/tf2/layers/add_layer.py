# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: add_layer.py
   Description : 
   Author : ericdoug
   date：2021/3/20
-------------------------------------------------
   Change Activity:
         2021/3/20: created
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
from tensorflow.keras.layers import Layer
import tensorflow as tf


# my packages


class AddLayer(Layer):
    def __init__(self, **kwargs):
        super(AddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(AddLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs,list):
            return inputs
        if len(inputs) == 1  :
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)