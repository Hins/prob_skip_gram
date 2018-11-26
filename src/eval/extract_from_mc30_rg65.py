# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 19:11
# @Author      : panxiaotong
# @Description : extract samples from MC-30 or RG-65 datasets

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("extract_from_mc30_rg65 <input file> <output file>")
        sys.exit()

    output_file = open(sys.argv[2], 'w')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            output_file.write(line.replace(';', '\t'))
        f.close()
    output_file.close()