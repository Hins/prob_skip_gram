# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 19:03
# @Author      : panxiaotong
# @Description : extract samples from SimLex-999

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_from_SL999 <input file> <output file>")
        sys.exit()

    output_file = open(sys.argv[2], 'w')
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('\t')
            word_1 = elements[0]
            word_2 = elements[1]
            score = elements[3]
            output_file.write(word_1 + "\t" + word_2 + "\t" + score + "\n")
        f.close()
    output_file.close()