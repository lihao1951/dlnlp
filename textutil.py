#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Name : textutil
Describe: 
    文本工具类
    need packages: jieba
Author : LH
Date : 2019/12/20
"""
import os
import jieba
from jieba import analyse


class WordUtil(object):
    """
    词语功能工具类
    wu = WordUtil(stopwords_path='stopwords', extend_dic_path='extenddic', vocabulary_idf_path='vocab')
    """

    def __init__(self, stopwords_path=None, extend_dic_path=None, vocabulary_idf_path=None):
        """
        构造函数
        :param stopwords_path: 停用词
        :param extend_dic_path: 自定义词典
        :param vocabulary_idf_path: 通用词典
        """
        # 定义<sos> <eos> <pad> <unk>字符
        self.start_token = '<sos>'
        self.end_token = '<eos>'
        self.pad_token = '<pad>'
        self.unknown_token = '<unk>'
        if extend_dic_path is not None:
            # 导入自定义词典
            jieba.load_userdict(os.path.abspath(extend_dic_path))
        if stopwords_path is not None:
            # 导入停用词词典
            self._stopwords = self._read_stopwords(os.path.abspath(stopwords_path))
        else:
            self._stopwords = {}
        if vocabulary_idf_path is not None:
            # 导入通用词典
            self._vocab, self._vocab_ix = self._read_vocabulary(os.path.abspath(vocabulary_idf_path))
        else:
            self._vocab = {}
        # 定义tfidf类，利用通用词典更新idf值
        self._tfidf = analyse.TFIDF(os.path.abspath(vocabulary_idf_path))

    @property
    def vocab(self):
        """
        返回通用词典
        :return:
        """
        return self._vocab

    @property
    def vocab_ix(self):
        """
        返回通用词典位置索引
        :return:
        """
        return self._vocab_ix

    @property
    def vocab_size(self):
        """
        获取通用词典的大小
        :return:
        """
        return len(self._vocab)

    def _read_stopwords(self, stopwords_path):
        """
        读取停用词表
        :param stopwords_path:
        :return:
        """
        stopwords = {}
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                stopwords[line] = 1
        return stopwords

    def _read_vocabulary(self, vocabulary_path):
        """
        读取通用词表
        返回词表及词表索引
        :param vocabulary_path:
        :return:
        """
        vocab = {}
        vocab_index = {}
        count = 0
        with open(vocabulary_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = line.split(' ')
                vocab[data[0]] = float(data[1])
                vocab_index[data[0]] = count
                count += 1
        # 添加<sos> <eos> <pad> <unk>字符
        vocab[self.start_token] = 1.0
        vocab[self.end_token] = 1.0
        vocab[self.pad_token] = 1.0
        vocab[self.unknown_token] = 1.0
        vocab_index[self.start_token] = count
        vocab_index[self.end_token] = count + 1
        vocab_index[self.pad_token] = count + 2
        vocab_index[self.unknown_token] = count + 3
        return vocab, vocab_index

    def cut(self, text):
        """
        一般分词
        :param text:
        :return: str
        """
        if text is not '':
            words = jieba.cut(text.lower())
            return ' '.join(words)
        return ''

    def cut_use_stopwords(self, text, top_k=None):
        """
        使用停用词表分词
        :param text:
        :return: str
        """
        if text is not '':
            words = self.cut(text).split(' ')
            words_result = []
            for word in words:
                if self._stopwords.__contains__(word):
                    continue
                words_result.append(word)
            if top_k:
                words_result = words_result[:top_k]
                while len(words_result) < top_k:
                    words_result.append(self.pad_token)
            return ' '.join(words_result)
        return ''

    def cut_2_id(self, text, top_k=None):
        """
        普通分词得到词表的id
        :param text:
        :param topN:
        :return:
        """
        if len(self._vocab) == 0:
            raise Exception('vocab is None')
        words = self.cut_use_stopwords(text, top_k).split(' ')
        ids = [self._vocab_ix.get(word, self._vocab_ix.get(self.unknown_token)) for word in words]
        return ids

    def cut_use_stopwords_vocab(self, text, top_k=None):
        """
        使用词典和停用词表分词
        :param text:
        :return: str
        """
        if len(self._vocab) == 0:
            raise Exception('vocab is None')
        if text is not '':
            words_result = []
            for word in self.cut_use_stopwords(text.lower(), top_k).split(' '):
                words_result.append(self.vocab.get(word, self.unknown_token))
            return ' '.join(words_result)
        return ''

    def cut_words2id(self, text, top_k=None):
        """
        使用词典和停用词表分词,并转化为id
        :param text:
        :return: str
        """
        indexes = []
        for word in self.cut_use_stopwords_vocab(text, top_k).split(' '):
            indexes.append(self._vocab_ix.get(word))
        return indexes

    def extract_keywords(self, text, top_k=10):
        """
        抽取关键词
        tfidf算法
        :param text:
        :param top_k:
        :return:
        """
        keywords = self._tfidf.extract_tags(self.cut_use_stopwords_vocab(text), topK=top_k)
        while len(keywords) < top_k:
            keywords.append(self.pad_token)
        return ' '.join(keywords)

    def extract_keywords2id(self, text, top_k=10):
        """
        抽取关键词 并将关键词转换为词表中的索引id
        :param text:
        :param top_k:
        :return: list
        """
        if len(self._vocab) == 0:
            raise Exception('vocab is None')
        words = self.extract_keywords(text, top_k).split(' ')
        indexes = []
        for word in words:
            ix = self._vocab_ix.get(word)
            indexes.append(ix)
        return indexes
