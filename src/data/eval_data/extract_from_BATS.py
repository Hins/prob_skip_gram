# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 15:09
# @Author      : panxiaotong
# @Description : extract train data from BATS for word analogy evaluation

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_BATS_samples <input file> <output file>")
        sys.exit()

    output_file = open(sys.argv[2], 'w')
    with open(sys.argv[1], 'r') as f:
        pre_line = ""
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                pre_line = line.strip('\r\n')
            else:
                output_file.write(line.strip('\r\n') + '\t' + pre_line + '\n')
    output_file.close()