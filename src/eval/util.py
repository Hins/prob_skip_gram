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
                     word_1,
                     word_2,
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
    :param word_1:
    :param word_2:
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
    sim_vec1 = []
    sim_vec2 = []
    word_emb_size = len(word_emb_dict[0])
    parser_emb_size = len(parser_emb_dict[0])
    kb_emb_size = len(kb_emb_dict[0])
    for info in info_list:
        if info == 'word':
            if word_1 not in word_dict or word_2 not in word_dict:
                return None, None
            sim_vec1.extend(word_emb_dict[word_dict[word_1]])
            sim_vec2.extend(word_emb_dict[word_dict[word_2]])
        elif info == 'pos':
            if word_1 not in pos_dict or word_2 not in pos_dict:
                return None, None
            sim_vec1.extend(pos_emb_dict[pos_dict[word_1]])
            sim_vec2.extend(pos_emb_dict[pos_dict[word_2]])
        elif info == 'parser':
            if word_1 not in parser_dict or word_2 not in parser_dict:
                return None, None
            emb1 = np.zeros(shape=[parser_emb_size], dtype=np.float32)
            for item in parser_dict[word_1]:
                emb1 = np.add(emb1, np.asarray(parser_emb_dict[item]))
            sim_vec1.extend(emb1.tolist())
            emb2 = np.zeros(shape=[parser_emb_size], dtype=np.float32)
            for item in parser_dict[word_2]:
                emb2 = np.add(emb2, np.asarray(parser_emb_dict[item]))
            sim_vec2.extend(emb2.tolist())
        elif info == 'dict':
            if word_1 not in dict_desc_dict or word_2 not in dict_desc_dict:
                return None, None
            emb1 = np.zeros(shape=[word_emb_size], dtype=np.float32)
            for item in dict_desc_dict[word_1]:
                emb1 = np.add(emb1, np.asarray(word_emb_dict[item]))
            sim_vec1.extend(emb1.tolist())
            emb2 = np.zeros(shape=[word_emb_size], dtype=np.float32)
            for item in dict_desc_dict[word_2]:
                emb2 = np.add(emb2, np.asarray(word_emb_dict[item]))
            sim_vec2.extend(emb2.tolist())
        elif info == 'kb':
            if word_1 not in kb_dict or word_2 not in kb_dict:
                return None, None
            emb1 = np.zeros(shape=[kb_emb_size], dtype=np.float32)
            for item in kb_dict[word_1]:
                emb1 = np.add(emb1, np.asarray(kb_emb_dict[item]))
            sim_vec1.extend(emb1.tolist())
            emb2 = np.zeros(shape=[kb_emb_size], dtype=np.float32)
            for item in kb_dict[word_2]:
                emb2 = np.add(emb2, np.asarray(kb_emb_dict[item]))
            sim_vec2.extend(emb2.tolist())
    return sim_vec1, sim_vec2