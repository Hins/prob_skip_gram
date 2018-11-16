# -*- coding: utf-8 -*-
# @Time       : 2018/3/5 14:13
# @Author     : panxiaotong
# @Desciption :  train w2v model

import os
import sys
import json
import random
import gensim

if __name__ == '__main__':
    if len(sys.argv) < 4:
        '''
        <separator>: separator
        <input file>: input file
        <model file>: output model file
        '''
        print("gensim_word2vec <separator> <input file> <model file>")
        sys.exit()

    separator = sys.argv[1]

    sentences = []
    input_file = open(sys.argv[2], 'r')
    for line in input_file:
        sentences.append([item for item in [item.replace('\n', '') for item in line.split(' ')]])
    input_file.close()

    model = gensim.models.Word2Vec(sentences, min_count=1, size=128, window=4, iter=15, negative=5, sample=1e-4)
    model.save(sys.argv[3])