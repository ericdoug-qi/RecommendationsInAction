# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: embedding_lookup.py
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


class EmbeddingLookup(Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config