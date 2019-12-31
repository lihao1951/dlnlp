#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : seq2seq
Describe: 
    
Author : LH
Date : 2019/12/27
"""
import tensorflow as tf
from tensorflow.contrib import seq2seq as s2s


class Config(object):
    batch_size = 32
    encode_embedding_size = 200
    decode_embedding_size = 200
    input_sequence_size = 100
    output_sequence_size = 20
    vocab_size = 50000


class SimpleSeq2Seq(object):
    def __init__(self, config):
        ...

    def train(self):
        # 构造输入
        # 编写encoder
        # 编写decoder
        # 输入数据 开始训练
        ...

    def inference(self):
        ...


tf.enable_eager_execution()
a = tf.constant(1, shape=[1])
p = tf.tile(a, [15])
print(p)
