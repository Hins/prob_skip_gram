# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 19:03
# @Author      : panxiaotong
# @Description : extract question type data from TREC dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("extract_from_trec <input_file> <word_dict> <output_file> <class_file>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()

    class_dict = {}
    with open(sys.argv[3], 'w') as out_f:
        with open(sys.argv[1], 'r') as in_f:
            for line in in_f:
                elements = line.strip('\r\n').split(':')
                if elements[0] not in class_dict:
                    class_dict[elements[0]] = len(class_dict)
                words = [item for item in elements[1].split(' ') if item.strip() != '']
                sent_ids = []
                for word in words:
                    sent_ids.append('0' if word not in word_dict else word_dict[word])
                if sent_ids.count('0') != len(sent_ids):
                    out_f.write(','.join(sent_ids) + '\t' + str(class_dict[elements[0]]) + '\n')
            in_f.close()
        out_f.close()

    with open(sys.argv[4], 'w') as f:
        for k,v in class_dict.items():
            f.write(k + '\t' + str(v) + '\n')
        f.close()
