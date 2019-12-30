#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : seq2seq
Describe: 
    
Author : LH
Date : 2019/12/27
"""
import tensorflow as tf
from tensorflow.contrib import seq2seq

class Config(object):
    batch_size = 32
    encode_embedding_size = 200
    decode_embedding_size = 200
    input_sequence_size = 100
    output_sequence_size = 20
    vocab_size = 50000

class SimpleSeq2Seq(object):
    def __init__(self,config):
        ...
