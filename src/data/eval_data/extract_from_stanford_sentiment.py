# -*- coding: utf-8 -*-
# @Time        : 2018/11/22 15:57
# @Author      : panxiaotong
# @Description : extract sentiment analysis data from Stanford sentiment Treebank dataset

import sys

if __name__ == "__main__":
    """
        :param input_file: dictionary.txt
        :param score_file: sentiment_labels.txt
    """
    if len(sys.argv) < 5:
        print("extract_sentiment_analysis_samples <input_file> <word_dict> <score_file> <output_file>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()

    score_dict = {}
    with open(sys.argv[3], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('|')
            score_dict[int(elements[0])] = float(elements[1])
        f.close()

    output_file = open(sys.argv[4], 'w')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('|')
            id = int(elements[1])
            try:
                tokenize_list = [str(item) for item in elements[0].split(' ') if item.strip() != ""]
            except:
                print(elements[0])
                continue
            id_list = []
            for token in tokenize_list:
                if token in word_dict:
                    id_list.append(word_dict[token])
                else:
                    id_list.append("0")
            if id_list.count("0") != len(id_list):
                output_file.write(','.join(id_list) + "\t" + str(score_dict[id]) + "\n")
        f.close()
    output_file.close()