# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: skip_gram.py
   Description : 
   Author : ericdoug
   date：2021/1/27
-------------------------------------------------
   Change Activity:
         2021/1/27: created
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
import tensorflow as tf

# my packages
from utils.file_helper import get_file_list, read_my_file_format

def process_data(my_path, batch_size=32, num_epochs=1):
    filenames = get_file_list(my_path)
    next_element = read_my_file_format(filenames, batch_size, num_epochs)
    return next_element

def get_session(gpu_fraction=0.1):
    """创建session，指定GPU或者CPU使用率
    
    :param gpu_fraction: 
    :return: 
    """

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)

    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class SkipGramModel(object):
    
    def __init__(self, 
                 vocab_size,
                 embed_size,
                 num_sampled,
                 train_optimizer,
                 learning_rate):
        
        self.vocab_size = vocab_size  # 词典长度
        self.embed_size = embed_size  # 词向量长度
        self.num_sampled = num_sampled  # 负采样数量
        self.train_optimizer = train_optimizer  # 优化方法
        self.learning_rate = learning_rate  # 学习率
        self.global_step = tf.Variable(0,
                                       dtype=tf.int32,
                                       trainable=False,
                                       name='global_step')
        
        
    def train(self,
              batch_data):
        """
        
        :param batch_data: 
        :return: 
        """
        
        # 定义输入数据
        with tf.name_scope("input_data"):
            # center words
            center_words = tf.reshape(batch_data['center_words'], shape=[-1])
            # target words
            target_words = tf.reshape(batch_data['target_words'], shape=[-1])

        # 定义网络输出
        with tf.name_scope("Compute_Score"):

            # 词向量矩阵
            with tf.variable_scope("embed", reuse=tf.AUTO_REUSE):
                self.embedding_dict = tf.get_variable(name='embed',
                                                      shape=[self.vocab_size, self.embed_size],
                                                      initializer=tf.glorot_uniform_initializer())

            # 模型内部参数矩阵
            with tf.variable_scope("nce", reuse=tf.AUTO_REUSE):
                self.nce_weight = tf.get_variable(name='nce_weight',
                                                  shape=[self.vocab_size, self.embed_size],
                                                  initializer=tf.glorot_uniform_initializer())

                self.nce_biases = tf.get_variable(name='nce_biased',
                                                  shape=[1],
                                                  initializer=tf.constant_initializer())

            # 将输入序列向量化
            # 其实就是一个简单的查表
            embed = tf.nn.embedding_lookup(self.embedding_dict, center_words, name='embed')

            # 得到NCE损失(负采样得到的损失)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,  # 权重
                    biases=self.nce_biases,  # 偏差
                    labels=target_words,  # 输入的标签
                    inputs=embed,  # 输入向量
                    num_sampled=self.num_sampled,  # 负采样的个数
                    num_classes=self.vocab_size  # 字典数目
                )
            )




        # 设定optimizer
        with tf.name_scope("optimizer"):
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):

                if self.train_optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                       beta1=0.9,
                                                       beta2=0.999,
                                                       epsilon=1e-8)
                elif self.train_optimizer == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                          initial_accumulator_value=1e-8)
                train_step = optimizer.minimize(loss, global_step=self.global_step)



        # 设定summary, 以便在Tensorboard里可视化
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("embedding_dict", self.embedding_dict)

            summary_op = tf.summary.merge_all()

        return train_step, loss, summary_op

