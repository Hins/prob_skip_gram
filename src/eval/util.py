# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 19:16
# @Author      : panxiaotong
# @Description : util functions

import numpy as np

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

def extract_features(use_other_info,
                     word,
                     word_dict,
                     word_emb_dict,
                     pos_dict,
                     pos_emb_dict,
                     parser_dict,
                     parser_emb_dict,
                     dict_desc_dict,
                     kb_dict,
                     kb_emb_dict):
    """
        :param use_other_info:
        :param word:
        :param word_dict:
        :param word_emb_dict:
        :param pos_dict:
        :param pos_emb_dict:
        :param parser_dict:
        :param parser_emb_dict:
        :param dict_desc_dict:
        :param kb_dict:
        :param kb_emb_dict:
        :return:
        """
    info_list = use_other_info.split(',')
    sim_vec = []
    word_emb_size = len(word_emb_dict[0])
    parser_emb_size = len(parser_emb_dict[0])
    kb_emb_size = len(kb_emb_dict[0])
    for info in info_list:
        if info == 'word':
            if word not in word_dict:
                return None
            sim_vec.extend(word_emb_dict[word_dict[word]])
        elif info == 'pos':
            if word not in pos_dict:
                return None
            sim_vec.extend(pos_emb_dict[pos_dict[word]])
        elif info == 'parser':
            if word not in parser_dict:
                return None
            emb = np.zeros(shape=[parser_emb_size], dtype=np.float32)
            for item in parser_dict[word]:
                emb = np.add(emb, np.asarray(parser_emb_dict[item]))
            sim_vec.extend(emb.tolist())
        elif info == 'dict':
            if word not in dict_desc_dict:
                return None
            emb = np.zeros(shape=[word_emb_size], dtype=np.float32)
            for item in dict_desc_dict[word]:
                emb = np.add(emb, np.asarray(word_emb_dict[item]))
            sim_vec.extend(emb.tolist())
        elif info == 'kb':
            if word not in kb_dict:
                return None
            emb = np.zeros(shape=[kb_emb_size], dtype=np.float32)
            for item in kb_dict[word]:
                emb = np.add(emb, np.asarray(kb_emb_dict[item]))
            sim_vec.extend(emb.tolist())
    return sim_vec