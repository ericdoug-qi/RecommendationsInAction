# _*_ coding: utf-8 _*_

"""
-------------------------------------------------
   File Name: train.py
   Description : 
   Author : ericdoug
   date：2021/3/15
-------------------------------------------------
   Change Activity:
         2021/3/15: created
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
import paddle
import paddle.nn.functional as F

# my packages
from baseline_model import BaselineModel

def train(model):
    # 配置训练参数
    lr = 0.001
    Epoches = 10
    paddle.set_device('cpu')

    # 启动训练
    model.train()
    # 获得数据读取器
    data_loader = model.train_loader
    # 使用adam优化器，学习率使用0.01
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

    for epoch in range(0, Epoches):
        for idx, data in enumerate(data_loader()):
            # 获得数据，并转为tensor格式
            usr, mov, score = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]
            scores_label = paddle.to_tensor(score)
            # 计算出算法的前向计算结果
            _, _, scores_predict = model(usr_v, mov_v)
            # 计算loss
            loss = F.square_error_cost(scores_predict, scores_label)
            avg_loss = paddle.mean(loss)

            if idx % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))

            # 损失函数下降，并清除梯度
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        # 每个epoch 保存一次模型
        paddle.save(model.state_dict(), './checkpoint/epoch' + str(epoch) + '.pdparams')

if __name__ == '__main__':
    fc_sizes = [128, 64, 32]
    data_root = "/Users/ericdoug/Documents/datas/ml-1m"
    user_info_file = os.path.join(data_root, "users.dat")
    movies_info_file = os.path.join(data_root, "movies.dat")
    user_rating_file = os.path.join(data_root, "ratings.dat")
    poster_rating_file = os.path.join(data_root, "new_rating.txt")
    poster_directory = os.path.join(data_root, "posters")
    use_poster, use_mov_title, use_mov_cat, use_age_job = False, True, True, True
    model = BaselineModel(user_info_file,
                          user_rating_file,
                          movies_info_file,
                          use_poster,
                          poster_directory,
                          poster_rating_file,
                          use_mov_title,
                          use_mov_cat,
                          use_age_job,
                          fc_sizes)
    train(model)