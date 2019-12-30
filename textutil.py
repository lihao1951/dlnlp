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
        if extend_dic_path is not None:
            # 导入自定义词典
            jieba.load_userdict(extend_dic_path)
        if stopwords_path is not None:
            # 导入停用词词典
            self._stopwords = self._read_stopwords(stopwords_path)
        else:
            self._stopwords = {}
        if vocabulary_idf_path is not None:
            # 导入通用词典
            self._vocab, self._vocab_ix = self._read_vocabulary(vocabulary_idf_path)
        else:
            self._vocab = {}
        # 定义tfidf类，利用通用词典更新idf值
        self._tfidf = analyse.TFIDF(vocabulary_idf_path)

    @property
    def vocab(self):
        """
        返回通用词典
        :return:
        """
        return self._vocab

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
        # 添加<sos> <eos> 字符
        vocab['<sos>'] = 1.0
        vocab['<eos>'] = 1.0
        vocab['<pad>'] = 1.0
        vocab_index['<sos>'] = count
        vocab_index['<eos>'] = count + 1
        vocab_index['<pad>'] = count + 2
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

    def cut_use_stopwords(self, text):
        """
        使用停用词表分词
        :param text:
        :return: str
        """
        if text is not '':
            words = jieba.cut(text.lower())
            words_result = []
            for x in words:
                if self._stopwords.__contains__(x):
                    continue
                words_result.append(x)
            return ' '.join(words_result)
        return ''

    def cut_use_stopwords_vocab(self, text, top_k=None):
        """
        使用词典和停用词表分词
        :param text:
        :return: str
        """
        if text is not '':
            words = self.cut_use_stopwords(text.lower())
            if len(self._vocab) == 0:
                if top_k is not None:
                    words = words.split(' ')[:top_k]
                    while len(words) < top_k:
                        words.append('<pad>')
                    return ' '.join(words)
                return words
            words_result = []
            for x in words.split(' '):
                if self._vocab.__contains__(x):
                    words_result.append(x)
            if top_k is not None:
                while len(words_result) < top_k:
                    words_result.append('<pad>')
                return ' '.join(words_result[:top_k])
            return ' '.join(words_result)
        return ''

    def cut_words2id(self, text, top_k=None):
        """
        使用词典和停用词表分词,并转化为id
        :param text:
        :return: str
        """
        words = self.cut_use_stopwords_vocab(text, top_k).split(' ')
        indexes = []
        for word in words:
            indexes.append(self._vocab_ix.get(word))
        return indexes

    def cut_words2id_with_tag(self, text, top_k=None):
        """
        使用词典和停用词表分词,并转化为id
        :param text:
        :return: str
        """
        indexes = self.cut_words2id(text, top_k)
        indexes.insert(0, self._vocab_ix.get('<sos>'))
        indexes.append(self._vocab_ix.get('<eos>'))
        return indexes

    def extract_keywords(self, text, top_k=10):
        """
        抽取关键词
        tfidf算法
        :param text:
        :param top_k:
        :return: str
        """
        keywords = self._tfidf.extract_tags(self.cut_use_stopwords_vocab(text), topK=top_k)
        while len(keywords) < top_k:
            keywords.append('<pad>')
        return ' '.join(keywords)

    def extract_keywords2id(self, text, top_k=10):
        """
        抽取关键词 并将关键词转换为词表中的索引id
        :param text:
        :param top_k:
        :return: list
        """
        words = self.extract_keywords(text, top_k).split(' ')
        indexes = []
        for word in words:
            # len(self.vocab)代表<unknown>标记
            ix = self._vocab_ix.get(word, len(self.vocab))
            indexes.append(ix)
        return indexes

    def extract_keywords2id_with_tag(self, text, top_k=10):
        """
        抽取关键词 并将关键词转换为词表中的索引id 并在首尾添加<sos> <eos>
        :param text:
        :param top_k:
        :return: list
        """
        indexes = self.extract_keywords2id(text, top_k)
        indexes.insert(0, self._vocab_ix.get('<sos>'))
        indexes.append(self._vocab_ix.get('<eos>'))
        return indexes
