# -*- coding: utf-8 -*-
# @Time        : 2018/11/21 14:32
# @Author      : panxiaotong
# @Description : extract negative samples

import random
import sys
sys.path.append("..")
from util.config import cfg
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("extract_negative_samples <target id> <context> <word dict> <output file>")
        sys.exit()

    target_list = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            target_list.append(line.strip('\r\n'))
        f.close()
    target_list = np.asarray(target_list, dtype=np.int32)

    context_list = []
    with open(sys.argv[2], 'r') as f:
        for line in f:
            context_list.append(line.strip('\r\n').split(','))
        f.close()
    context_list = np.asarray(context_list, dtype=np.int32)

    word_dictionary_size = len(open(sys.argv[3]).readlines()) + 1

    neg_list = np.zeros(shape=[context_list.shape[0], cfg.negative_sample_size], dtype=np.int32)
    for index, words in enumerate(context_list):
        sub_labels = np.full(shape=[cfg.negative_sample_size],
                             fill_value=word_dictionary_size + 1, dtype=np.int32)
        iter = 0
        while iter < cfg.negative_sample_size:
            r = random.randint(0, word_dictionary_size - 1)
            flag = False
            for word in sub_labels:
                if r == word:
                    flag = True
                    break
            if flag == True:
                continue
            for word in words:
                if r == word:
                    flag = True
                    break
            if flag == False:
                sub_labels[iter] = r
                iter += 1
        np.random.shuffle(sub_labels)  # shuffle positive and negative samples
        neg_list[index] = sub_labels
    np.savetxt(sys.argv[4], neg_list, fmt="%s", delimiter=',')