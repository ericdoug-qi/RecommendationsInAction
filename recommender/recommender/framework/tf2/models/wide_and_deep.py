# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: wide_and_deep.py
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
import sys

# third packages
import tensorflow as tf

# my packages

# tf.compat.v1.enable_v2_behavior()

ROOT_PATH = '/Users/ericdoug/Documents/datas/adult/'
#TRAIN_PATH = ROOT_PATH + 'train.csv'
TRAIN_PATH = ROOT_PATH + 'adult.data'
EVAL_PATH = ROOT_PATH + 'adult.test'
PREDICT_PATH = ROOT_PATH + 'predict.csv'
MODEL_PATH = 'adult_model'
EXPORT_PATH = 'adult_export_model'
# _CSV_COLUMNS = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education_num',
#     'marital_status', 'occupation', 'relationship', 'race', 'gender',
#     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
#     'income_bracket'
# ]

_CSV_COLUMNS = [                                #定义CVS列名
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_area', 'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [                        #定义每一列的默认值
        [0], [''], [0], [''], [0],
        [''], [''], [''], [''], [''],
        [0], [0], [0], [''], ['']]

# _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
#                         [0], [0], [0], [''], [0]]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations.
    # age_buckets = tf.feature_column.bucketized_column(
    #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def parse_csv(value):
    columns = tf.io.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    classes = tf.equal(labels, '>50K')  # binary classification
    return features, classes

def input_fn(data_path, shuffle, num_epochs, batch_size):
    """Generate an input function for the Estimator."""
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_path)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    # for item in dataset.take(1):
    #     parsed_output = tf.io.decode_csv(item, _CSV_COLUMN_DEFAULTS)
    #     features = dict(zip(_CSV_COLUMNS, parsed_output))
    #     labels = features.pop('income_bracket')
    #     print(f"features: {features}, labels:{labels}")
    #     print(features['age'].numpy())
    #     print(features['workclass'].numpy())
    #     print(tf.equal(labels, ">50K"))

    dataset = dataset.map(parse_csv)


    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


# estimator.train()可以循环运行，模型的状态将持久保存在model_dir
def run():
    wide_columns, deep_columns = build_model_columns()

    # os.system('rm -rf {}'.format(MODEL_PATH))
    config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    estimator = tf.estimator.DNNLinearCombinedClassifier(model_dir=MODEL_PATH,
                                                         linear_feature_columns=wide_columns,
                                                         linear_optimizer=tf.keras.optimizers.Ftrl(learning_rate=0.01),
                                                         dnn_feature_columns=deep_columns,
                                                         dnn_hidden_units=[256, 64, 32, 16],
                                                         dnn_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                                         # loss_reduction=tf.keras.losses.Reduction.SUM,
                                                         config=config)
    # Linear model.
    # estimator = tf.estimator.LinearClassifier(feature_columns=wide_columns, n_classes=2,
    #                                           optimizer=tf.train.FtrlOptimizer(learning_rate=0.03))

    # Train the model.
    estimator.train(
        input_fn=lambda: input_fn(data_path=TRAIN_PATH, shuffle=True, num_epochs=40, batch_size=100), steps=2000)
    """
    steps: 最大训练次数，模型训练次数由训练样本数量、num_epochs、batch_size共同决定，通过steps可以提前停止训练
    """
    # Evaluate the model.
    eval_result = estimator.evaluate(
        input_fn=lambda: input_fn(data_path=EVAL_PATH, shuffle=False, num_epochs=10, batch_size=40))

    print('Test set accuracy:', eval_result)

    # Predict.
    # pred_dict = estimator.predict(
    #     input_fn=lambda: input_fn(data_path=PREDICT_PATH, shuffle=False, num_epochs=1, batch_size=40))
    # for pred_res in pred_dict:
    #     print(pred_res['probabilities'][1])

    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(EXPORT_PATH, serving_input_fn)


if __name__ == '__main__':
    run()

