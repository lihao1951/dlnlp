#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : seq2seq
Describe: 
    
Author : LH
Date : 2019/12/27
"""
import tensorflow as tf
from log import get_logger

logger = get_logger(__file__)


class SimpleSeq2Seq(object):
    def __init__(self, model_path=None, config=None):
        if model_path:
            self.load_model(model_path)
            logger.info('load model ' + model_path)
            return
        if config is None:
            logger.error('config is None')
            raise Exception('config is None')
        self._src_vocab_size = config.src_vocab_size
        self._tgt_vocab_size = config.tgt_vocab_size
        self._encode_embedding_size = config.encode_embedding_size
        self._decode_embedding_size = config.decode_embedding_size
        self._start_token_id = config.start_token_id
        self._end_token_id = config.end_token_id
        self._share_embedding = config.share_embedding
        self._cell_type = config.cell_type.lower()
        self._cell_layer_size = config.cell_layer_size
        self._cell_layer_units = config.cell_layer_units
        self._bi_direction = config.bi_direction
        self._learning_rate = config.learning_rate
        logger.info('start build model')

    def train(self, generator, model_path):
        # 构造计算图
        if model_path is None:
            raise FileNotFoundError('model path is not found')
        graph = self._build_graph()
        with tf.Session(graph=graph) as sess:
            # 模型保存
            saver = tf.train.Saver()
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            # 输入数据 开始训练
            for src_data, tgt_data_x, tgt_data_y, src_data_len, tgt_data_max_len, tgt_data_len, batch_size, epoch in generator:
                train_loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                    self.src_data: src_data,
                    self.src_data_len: src_data_len,
                    self.tgt_data_x: tgt_data_x,
                    self.tgt_data_y: tgt_data_y,
                    self.tgt_data_len: tgt_data_len,
                    self.tgt_data_max_len: tgt_data_max_len
                })
                logger.info('epoch: {} batch:{} train-loss:{}'.format(epoch, batch_size, train_loss))
            saver.save(sess, model_path)

    def _build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # 构造输入
            self._build_input()
            # 编码器
            encode_outputs, encode_states = self._build_encode()
            # 解码器
            decode_output = self._build_decode(encode_states)
            training_logits = tf.identity(decode_output.rnn_output, 'logits')
            masks = tf.sequence_mask(self.tgt_data_len, self.tgt_data_max_len, dtype=tf.float32, name='masks')
            with tf.name_scope('optimization'):
                self.loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.tgt_data_y, masks)
                optimizer = tf.train.AdamOptimizer(self._learning_rate)
                gradients = optimizer.compute_gradients(self.loss)
                clip_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(clip_gradients)
        return graph

    def get_rnn_layer(self, num_units):
        """
        获得rnn单元层
        :return:
        """

        def _rnn_cell(cell_type, num_units):
            """
            内部函数
            :param cell_type:
            :param num_units:
            :return:
            """
            if cell_type == 'gru':
                return tf.nn.rnn_cell.GRUCell(num_units=num_units)
            else:
                return tf.nn.rnn_cell.LSTMCell(num_units=num_units)

        return tf.nn.rnn_cell.MultiRNNCell(
            [_rnn_cell(self._cell_type, num_units) for _ in range(self._cell_layer_size)])

    def _bi_encode(self):
        """
        双向编码层
        :return:
        """
        with tf.name_scope('bi-encode'):
            fw_rnn = self.get_rnn_layer(self._cell_layer_units)
            bw_rnn = self.get_rnn_layer(self._cell_layer_units)
            # 下面进行双向展开
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(fw_rnn, bw_rnn, self.src_encode_data,
                                                                    sequence_length=self.src_data_len, dtype=tf.float32)
            encoder_outputs = tf.concat(bi_outputs, 2)
            fw_encode_state = bi_states[0]
            bw_encode_state = bi_states[1]
            encode_states = []
            for i in range(self._cell_layer_size):
                if self._cell_type == 'gru':
                    # 每层gru输出只有一个值
                    s = tf.concat([fw_encode_state[i], bw_encode_state[i]], -1)
                    encode_states.append(s)
                else:
                    # 每层lstm输出有两个值c h
                    c = tf.concat([fw_encode_state[i][0], bw_encode_state[i][0]], -1)
                    h = tf.concat([fw_encode_state[i][1], bw_encode_state[i][1]], -1)
                    encode_states.append(tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c))
            encode_states = tuple(encode_states)
            return encoder_outputs, encode_states

    def _single_encode(self):
        """
        单向编码层
        :return:
        """
        with tf.name_scope('single-encode'):
            rnn_cell = self.get_rnn_layer(self._cell_layer_units)
            # 下面进行单向展开
            # states 是一个tuple
            outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.src_encode_data, sequence_length=self.src_data_len,
                                                dtype=tf.float32)
            return outputs, states

    def _build_encode(self):
        if self._bi_direction:
            encode_outputs, encode_states = self._bi_encode()
        else:
            encode_outputs, encode_states = self._single_encode()
        return encode_outputs, encode_states

    def _build_decode(self, states):
        with tf.name_scope('decode-rnn'):
            if self._cell_type == 'gru':
                self.decode_rnn_layer = self.get_rnn_layer(self._cell_layer_units)
            else:
                self.decode_rnn_layer = self.get_rnn_layer(self._cell_layer_units * 2)
            with tf.name_scope("decoder-porjection"):
                # 为什么这里projection_layer不指定激活函数为softmax，最后构建loss的传入的是logits，我的理解logits是没有经过激活函数的
                # 的值，logits = W*X+b
                self.projection_layer = tf.layers.Dense(units=self._tgt_vocab_size,
                                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                           stddev=0.1))
            # 训练使用的training_helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                self.tgt_decode_data_x, self.tgt_data_len, time_major=False)
            # 定义decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decode_rnn_layer, training_helper, states,
                output_layer=self.projection_layer)
            decode_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, swap_memory=True)
            return decode_output

    def _build_input(self):
        with tf.name_scope('input-tensor'):
            # 编码层输入数据
            self.src_data = tf.placeholder(tf.int32, shape=[None, None], name='source-data')
            # 编码层输入数据长度
            self.src_data_len = tf.placeholder(tf.int32, shape=[None], name='source-data-len')
            # 解码层输入数据
            self.tgt_data_x = tf.placeholder(tf.int32, shape=[None, None], name='target-data-x')
            # 解码层输出数据
            self.tgt_data_y = tf.placeholder(tf.int32, shape=[None, None], name='target-data-y')
            # 解码层数据长度
            self.tgt_data_len = tf.placeholder(tf.int32, shape=[None], name='target-data-len')
            # 编码层词嵌入
            src_embedding = tf.get_variable(name='src-embedding',
                                            shape=[self._src_vocab_size, self._encode_embedding_size])
            self.src_encode_data = tf.nn.embedding_lookup(src_embedding, self.src_data)
            # 是否共享词嵌入矩阵，得到解码层词嵌入输入
            if self._share_embedding:
                self.tgt_decode_data_x = tf.nn.embedding_lookup(src_embedding, self.tgt_data_x)
            else:
                tgt_embedding = tf.get_variable(name='tgt-embedding',
                                                shape=[self._tgt_vocab_size, self._encode_embedding_size])
                self.tgt_decode_data_x = tf.nn.embedding_lookup(tgt_embedding, self.tgt_data_x)
            # 保存当前batch的最长序列值，mask的时候需要用到
            self.tgt_data_max_len = tf.placeholder(tf.int32, shape=[], name='target-data-max-len')
            # 测试时输入数据的batch是多少，动态的传入，避免预测时必须固定batch
            predict_batch_size = tf.placeholder(dtype=tf.int32, shape=[1], name="input_batch_size")
            # tf.tile将常量在同一维度重复shape次数连接在一起
            self.start_tokens = tf.tile(tf.constant(value=self._start_token_id, dtype=tf.int32, shape=[1]),
                                        multiples=predict_batch_size, name="start_tokens")

    def inference(self):
        ...

    def load_model(self, model_path):
        ...
