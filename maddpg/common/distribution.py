# -*- coding: utf-8 -*-
# @Date       : 2024/5/2417:31
# @Auther     : Wang.zr
# @File name  : distribution.py
# @Description:
import torch as th


def gen_action_for_discrete(actions):
    # u = tf.random.uniform(tf.shape(actions), dtype=tf.float64)
    u = th.rand_like(actions)
    return th.nn.functional.softmax(actions - th.log(-th.log(u)), dim=-1)


def gen_action_for_continuous(actions):
    mean, logstd = th.split(actions, 2, dim=1)
    # 计算标准差,即将对数标准差进行指数运算
    std = th.exp(logstd)
    # 生成与均值形状相同的随机正态分布张量,并与标准差相乘得到最终的动作
    # 这相当于TensorFlow实现中的mean + std * tf.random.normal(tf.shape(mean))
    return mean + std * th.randn_like(mean)
