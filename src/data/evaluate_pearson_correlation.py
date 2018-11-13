# -*- coding: utf-8 -*-
# @Time        : 2018/11/13 10:45
# @Author      : panxiaotong
# @Description : evaluate by pearson correlation

import numpy as np
import operator
import sys
import scipy.stats as stats

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

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("evaluate_pearson_correlation <word_emb> <word_dict> <evaluation_ds>")
        sys.exit()

    word_emb_list = {}
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            elements = line.strip('\r\n').split(',')
            word_emb_list[idx + 1] = [float(item) for item in elements]
        f.close()

    word_dict = {}
    reverse_word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = int(elements[1])
            reverse_word_dict[int(elements[1])] = elements[0]
        f.close()

    WORD_SEPARATOR = '_'
    word_pair_dict = {}
    index_dict = {}
    pred_score_dict = {}
    with open(sys.argv[3], 'r') as f:
        counter = 0
        real_counter = 0
        for idx, line in enumerate(f):
            elements = line.strip('\r\n').split(' ')
            word_1 = elements[0].split('-')[0]
            word_2 = elements[1].split('-')[1]
            score = float(elements[2])
            word_pair_dict[word_1 + WORD_SEPARATOR + word_2] = score
            if word_1 in word_dict and word_2 in word_dict:
                pred_score_dict[counter] = cosin_distance(word_emb_list[word_dict[word_1]],
                                                          word_emb_list[word_dict[word_2]])
                index_dict[counter] = score
                counter += 1
            real_counter += 1
        print(real_counter)
        f.close()
    print(len(pred_score_dict))
    print(len(index_dict))
    sorted_x = sorted(index_dict.items(), key=operator.itemgetter(1))
    real_value_list = []
    for k,v in sorted_x:
        real_value_list.append(k)
    sorted_x = sorted(pred_score_dict.items(), key=operator.itemgetter(1))
    pred_value_list = []
    for k,v in sorted_x:
        pred_value_list.append(k)
    print(stats.pearsonr(real_value_list, pred_value_list)[0])