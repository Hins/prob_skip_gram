# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 18:42
# @Author      : panxiaotong
# @Description : extract samples from MEN

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_from_MEN <input file> <output file>")
        sys.exit()

    output_file = open(sys.argv[2], 'w')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(' ')
            word_1 = elements[0].split('-')[0]
            word_2 = elements[1].split('-')[0]
            score = elements[2]
            output_file.write(word_1 + "\t" + word_2 + "\t" + score + "\n")
        f.close()
    output_file.close()