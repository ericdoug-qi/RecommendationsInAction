# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: fm.py
   Description : 
   Author : ericdoug
   date：2021/3/19
-------------------------------------------------
   Change Activity:
         2021/3/19: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os
import datetime

# third packages
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from collections import namedtuple, OrderedDict


# my packages
from recommender.recommender.models.tf2.layers.embedding_lookup import EmbeddingLookup


SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed','embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed','reduce_type','dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size','hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim','maxlen', 'dtype'])
# 筛选实体标签categorical 用于定义映射关系
DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                    "keyword_id": [str(i) for i in range(0, 100)],
           }

feature_columns = [SparseFeat(name="topic_id", voc_size=700, hash_size= None,share_embed=None, embed_dim=8, dtype='int32'),
                   SparseFeat(name="keyword_id", voc_size=10, hash_size= None,share_embed=None, embed_dim=8, dtype='int32'),
                   SparseFeat(name='client_type', voc_size=2, hash_size= None,share_embed=None, embed_dim=8,dtype='int32'),
                   SparseFeat(name='post_type', voc_size=2, hash_size= None,share_embed=None, embed_dim=8,dtype='int32'),
                   VarLenSparseFeat(name="follow_topic_id", voc_size=700, hash_size= None,share_embed='topic_id',weight_name = None, combiner= 'sum', embed_dim=8, maxlen=20,dtype='int32'),
                   VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, hash_size= None,share_embed='topic_id', weight_name = 'all_topic_fav_7_weight', combiner= 'sum', embed_dim=8, maxlen=5,dtype='int32'),
                   ]


DEFAULT_VALUES = [[0],[''],[0.0],[0.0], [0.0],
                  [''], [''],[0.0]]
COL_NAME = ['act', 'client_id', 'client_type', 'post_type', 'topic_id', 'follow_topic_id', 'all_topic_fav_7', 'keyword_id']



def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


def build_embedding_matrix(features_columns, linear_dim=None):
    """构造 自定义embedding层 matrix

    :param features_columns:
    :param linear_dim:
    :return:
    """
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            vocab_size = feat_col.voc_size + 2
            embed_dim = feat_col.embed_dim if linear_dim is None else 1
            name_tag = '' if linear_dim is None else '_linear'
            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim),mean=0.0,
                                                                           stddev=0.001, dtype=tf.float32), trainable=True, name=vocab_name+'_embed'+name_tag)
    return embedding_matrix


def build_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns)

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name)

    return embedding_dict


def build_linear_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns, linear_dim=1)
    name_tag = '_linear'

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name + name_tag)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name + name_tag)

    return embedding_dict


def input_from_feature_columns(features, features_columns, embedding_dict):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys)(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)

            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys, mask_value='0')(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)
            if feat_col.combiner is not None:
                input_sparse = DenseToSparseTensor(mask_value=0)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)

            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])

        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise("dnn_feature_columns can not be empty list")


def get_linear_logit(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add()([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        return dense_linear_layer
    else:
        raise("linear_feature_columns can not be empty list")


def FM(feature_columns):
    """Instantiates the FM Network architecture.
        Args:
            feature_columns: An iterable containing all the features used by fm model.
        return: A Keras model instance.
        """
    features = build_input_features(feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    sparse_varlen_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    inputs_list = list(features.values())

    # 构建 linear embedding_dict
    linear_embedding_dict = build_linear_embedding_dict(feature_columns)
    linear_sparse_embedding_list, linear_dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                                       linear_embedding_dict)
    # linear part
    linear_logit = get_linear_logit(linear_sparse_embedding_list, linear_dense_value_list)

    # 构建 embedding_dict
    cross_columns = sparse_feature_columns + sparse_varlen_feature_columns
    embedding_dict = build_embedding_dict(cross_columns)
    sparse_embedding_list, _ = input_from_feature_columns(features, cross_columns, embedding_dict)

    # 将所有sparse的k维embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_embedding_list)  # ?, n, k
    # cross part
    fm_cross_logit = FMLayer()(concat_sparse_kd_embed)

    final_logit = Add()([fm_cross_logit, linear_logit])

    output = tf.keras.layers.Activation("sigmoid", name="fm_out")(final_logit)
    model = Model(inputs=inputs_list, outputs=output)

    return model



