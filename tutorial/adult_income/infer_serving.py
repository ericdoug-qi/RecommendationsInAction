# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: infer_serving.py
   Description : 
   Author : ericdoug
   date：2021/3/21
-------------------------------------------------
   Change Activity:
         2021/3/21: created
-------------------------------------------------
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# sys packages
import os
import time

# third packages
import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# my packages

ROOT_PATH = '/Users/ericdoug/Documents/datas/adult/'
#TRAIN_PATH = ROOT_PATH + 'train.csv'
TRAIN_PATH = ROOT_PATH + 'adult.data'
EVAL_PATH = ROOT_PATH + 'adult.test'
TEST_PATH = ROOT_PATH + 'test.csv'

_CSV_COLUMNS = [                                #定义CVS列名
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_area', 'income_bracket'
]


_CSV_COLUMN_DEFAULTS = [                        #定义每一列的默认值
        'int', 'string', 'int', 'string', 'int',
        'string', 'string', 'string', 'string', 'string',
        'int', 'int', 'int', 'string', 'string']

# 生成tf.Example数据
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

feature_dict = {}
serialized_strings = []

with open(TEST_PATH, encoding='utf-8') as f:
    lines = f.readlines()

    # types = [key for key in lines[0].strip('\n').split(',')]

    for i in range(0, len(lines)):
        items = [key for key in lines[i].strip().split(',')]
        for j in range(len(items)):
            item = items[j]
            if _CSV_COLUMN_DEFAULTS[j] == 'int':
                item = int(item)
                feature_dict[_CSV_COLUMNS[j]] = _float_feature(item)
            elif _CSV_COLUMN_DEFAULTS[j] == 'string':
                feature_dict[_CSV_COLUMNS[j]] = _bytes_feature(bytes(item, encoding='utf-8'))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example_proto.SerializeToString()
        serialized_strings.append(serialized)

channel = grpc.insecure_channel(target='localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'adult_export_model'
request.model_spec.signature_name = 'predict'

data = serialized_strings
size = len(data)
request.inputs['examples'].CopyFrom(tf.compat.v1.make_tensor_proto(data, shape=[size]))
start_time = time.time()
pred_dict = stub.Predict(request, 10.0)
print(pred_dict)
for pred_res in pred_dict:
    print(pred_res['probabilities'][1])
print(f"Time cost: {time.time() - start_time}")


