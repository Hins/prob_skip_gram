# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 15:16
# @Author      : panxiaotong
# @Description : extract train data from Google analogy test set for word analogy evaluation

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_google_samples <input file> <output file>")
        sys.exit()

    output_file = open(sys.argv[2], 'w')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            if line.find(":") != -1:
                continue
            output_file.write(line.strip('\r\n').replace(' ', '\t') + '\n')
    output_file.close()