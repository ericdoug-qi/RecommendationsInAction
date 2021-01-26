# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: nrms.py
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
import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
from tempfile import TemporaryDirectory
import tensorflow as tf
import papermill as pm
import scrapbook as sb

# third packages
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

# my packages
from configs.nrms_setting import data_root
from configs.nrms_setting import MIND_type
from configs.nrms_setting import batch_size
from configs.nrms_setting import epochs
from configs.nrms_setting import seed
from utils.log import get_logger
from configs.nrms_setting import BASE_DIR


log_file = os.path.join(BASE_DIR, "logs", "nrms.log")

logger = get_logger(log_file)

train_news_file = os.path.join(data_root, 'train', r'news.tsv')
logger.debug(f"train_news_file: {train_news_file}")
train_behaviors_file = os.path.join(data_root, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_root, 'dev', r'news.tsv')
valid_behaviors_file = os.path.join(data_root, 'dev', r'behaviors.tsv')
test_news_file = os.path.join(data_root, 'test', r'news.tsv')
test_behaviors_file = os.path.join(data_root, 'test', r'behaviors.tsv')
wordEmb_file = os.path.join(data_root, "utils", "embedding.npy")
userDict_file = os.path.join(data_root, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_root, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_root, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_root, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url,
                               os.path.join(data_root, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/',
                               os.path.join(data_root, 'utils'), mind_utils)


hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
logger.debug(f"hparams: {hparams}")

iterator = MINDIterator

model = NRMSModel(hparams, iterator, seed=seed)

logger.info(model.run_eval(valid_news_file, valid_behaviors_file))


model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
logger.debug(f"res_syn: {res_syn}")

sb.glue("res_syn", res_syn)

model_path = os.path.join(BASE_DIR, "ckpt")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_ckpt"))

group_impr_indexes, group_labels, group_preds = model.run_fast_eval(test_news_file, test_behaviors_file)

with open(os.path.join(BASE_DIR, 'submits', 'prediction.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')


f = zipfile.ZipFile(os.path.join(BASE_DIR, 'submits', 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(BASE_DIR, 'submits', 'prediction.txt'), arcname='prediction.txt')
f.close()
