# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 18:44
# @Author      : panxiaotong
# @Description : extract movie review sentiment test data from mc dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("extract_from_movie_review_test <test_file> <score_file> <word_dict> <output_file>")
        sys.exit()

    score_dict = {}
    with open(sys.argv[2], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split(',')
            score_dict[elements[0]] = elements[1]
        f.close()

    word_dict = {}
    with open(sys.argv[3], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()

    output_file = open(sys.argv[4], 'w')
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('\t')
            words = [item for item in elements[2].split(' ') if item.strip() != '']
            sent_ids = []
            for word in words:
                sent_ids.append('0' if word not in word_dict else word_dict[word])
            if sent_ids.count('0') != len(sent_ids):
                output_file.write(','.join(sent_ids) + '\t' + score_dict[elements[0]] + '\n')
        f.close()
    output_file.close()