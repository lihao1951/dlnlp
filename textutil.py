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
    def __init__(self, stopwords_path=None, extend_dic_path=None, vocabulary_idf_path=None):
        if extend_dic_path is not None:
            jieba.load_userdict(extend_dic_path)
        if stopwords_path is not None:
            self._stopwords = self._read_stopwords(stopwords_path)
        else:
            self._stopwords = {}
        if vocabulary_idf_path is not None:
            self._vocab = self._read_vocabulary(vocabulary_idf_path)
        else:
            self._vocab = {}
        # 定义idf类，给出词的idf值
        self._tfidf = analyse.TFIDF(vocabulary_idf_path)

    @property
    def vocab(self):
        return self._vocab

    @classmethod
    def _read_stopwords(cls, stopwords_path):
        stopwords = {}
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                stopwords[line] = 1
        return stopwords

    @classmethod
    def _read_vocabulary(cls, vocabulary_path):
        vocab = {}
        with open(vocabulary_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
            for ww in words.strip().split():
                vocab[ww[0]] = float(ww[1])
        return vocab

    def cut(self, text):
        if text is not '':
            words = jieba.cut(text)
            return ' '.join(words)
        return ''

    def cut_with_stopwords(self, text):
        if text is not '':
            words = jieba.cut(text)
            words_result = []
            for x in words:
                if self._stopwords.__contains__(x):
                    continue
                words_result.append(x)
            return ' '.join(words_result)
        return ''

    def cut_with_stop_words_vocab(self, text):
        if text is not '':
            words = self.cut_with_stopwords(text)
            words_result = []
            for x in words.split(' '):
                if self._vocab.__contains__(x) and len(self._vocab) > 0:
                    words_result.append(x)
            if len(words_result) == 0:
                return words
            return ' '.join(words_result)
        return ''

    def extract_keywords(self, text, top_k=10):
        keywords = self._tfidf.extract_tags(self.cut_with_stopwords(text), topK=top_k)
        return ' '.join(keywords)


text = '北京时间12月19日凌晨，在万众瞩目下迎来的西班牙国家德比战中，皇马客场0：0逼平巴萨，' \
       '派出最强阵容的两支豪门在90分钟内踢出了最无聊的一场国家德比。梅西2次射门、1次射正，' \
       '并有46次传球。全场比赛，梅西没有破门。90分钟闷平之后的皇马也开始想念C罗，后者在北京时' \
       '间今天凌晨对阵桑普多利亚的比赛中打入了一记不可思议的滞空头球，简直逆天。昨夜今晨，欧洲' \
       '足坛迎来了备受瞩目的一场比赛，巴萨坐镇主场迎战皇马，双方均排出了几乎最强的阵容，梅西、' \
       '苏亚雷斯、格列兹曼、本泽马、贝尔等群星全部登场，两队总身价超过了185亿人民币。'

wu = WordUtil(stopwords_path='stopwords', extend_dic_path='extenddic')
wu.extract_keywords(text)
