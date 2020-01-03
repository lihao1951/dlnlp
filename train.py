#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Name train
@Description
    
@Author LiHao
@Date 2020/1/3
"""
import re
import numpy as np
from textutil import WordUtil

from seq2seq import SimpleSeq2Seq

wu = WordUtil(stopwords_path='stopwords', extend_dic_path='extenddic', vocabulary_idf_path='vocab')


class Config(object):
    encode_embedding_size = 50  # 编码词嵌入大小
    decode_embedding_size = 50  # 解码词嵌入大小
    src_vocab_size = wu.vocab_size  # 编码层词典大小
    tgt_vocab_size = wu.vocab_size  # 解码层词典大小
    start_token_id = wu.vocab_ix.get(wu.start_token)  # 开始token id
    end_token_id = wu.vocab_ix.get(wu.end_token)  # 结尾token id
    share_embedding = False  # 是否共享嵌入层
    cell_type = 'lstm'  # rnn 单元type
    cell_layer_size = 2  # rnn层数
    cell_layer_units = 16  # rnn单元数目
    bi_direction = True  # 双向
    learning_rate = 0.001


def get_generator(src_len, tgt_len, batch_size, epochs):
    for epoch in range(epochs):
        with open('original-data.txt', 'r', encoding='utf-8') as f:
            size = 0
            src_data_list = []
            src_data_len_list = []
            tgt_data_x_list = []
            tgt_data_y_list = []
            tgt_data_len_list = []
            for line in f:
                size += 1
                line = line.strip()
                title, cont = re.split(r'-\*-', line)
                src_data = wu.cut_2_id(cont, src_len)
                tgt_data_x = wu.cut_2_id(title)
                tgt_data_y = wu.cut_2_id(title)
                tgt_data_x = tgt_data_x[:tgt_len]
                tgt_data_x.insert(0, wu.vocab_ix.get(wu.start_token))
                tgt_data_len_list.append(len(tgt_data_x))
                # 补全x
                while len(tgt_data_x) < tgt_len + 1:
                    tgt_data_x.append(wu.vocab_ix.get(wu.pad_token))
                # 补全y
                tgt_data_y = tgt_data_y[:tgt_len]
                tgt_data_y.append(wu.vocab_ix.get(wu.end_token))
                while len(tgt_data_y) < tgt_len + 1:
                    tgt_data_y.append(wu.vocab_ix.get(wu.pad_token))
                src_data_list.append(src_data)
                src_data_len_list.append(len(src_data))
                tgt_data_x_list.append(tgt_data_x)
                tgt_data_y_list.append(tgt_data_y)
                if size == batch_size:
                    tgt_data_max_len = np.max(tgt_data_len_list)
                    yield np.array(src_data_list, dtype=np.int), np.array(tgt_data_x_list, dtype=np.int), np.array(
                        tgt_data_y_list,
                        dtype=np.int), src_data_len_list, tgt_data_max_len, tgt_data_len_list, batch_size, epoch
                    src_data_list.clear()
                    tgt_data_x_list.clear()
                    tgt_data_y_list.clear()
                    src_data_len_list.clear()
                    tgt_data_len_list.clear()
                    size = 0
            if size > 0:
                tgt_data_max_len = np.max(tgt_data_len_list)
                yield np.array(src_data_list, dtype=np.int), np.array(tgt_data_x_list, dtype=np.int), np.array(
                    tgt_data_y_list, dtype=np.int), src_data_len_list, tgt_data_max_len, tgt_data_len_list, size, epoch


config = Config()
gen = get_generator(100, 10, 16, 10)
model = SimpleSeq2Seq(config=config)
model.train(gen, './models/s2s.ckpt')
