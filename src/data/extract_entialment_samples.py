# -*- coding: utf-8 -*-
# @Time        : 2018/11/23 10:45
# @Author      : panxiaotong
# @Description : extract entailment samples from SICK dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("extract_entailment_samples <input file> <word dict> <output file1> <output file2>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()

    output_single = open(sys.argv[3], 'w')
    output_complex = open(sys.argv[4], 'w')
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('\t')
            sent1 = elements[1]
            sent2 = elements[2]
            class_single = elements[3]
            score = float(elements[4])
            
        f.close()

    output_single.close()
    output_complex.close()