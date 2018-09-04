# -*- coding: utf-8 -*-
# @Time        : 2018/9/4 18:43
# @Author      : panxiaotong
# @Description : extract word id and context info

import sys
sys.path.append("..")
from util.config import cfg

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("<extract_word_id> <input file> <word dict output file> <word id output file> <context output file>")
        sys.exit()

    word_dict = {}
    word_list = []
    context_list = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(",")

        f.close()