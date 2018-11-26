# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 11:22
# @Author      : panxiaotong
# @Description : evaluate by nearest neighbor, qualitative evaluation

import sys

from util import extract_features, cosin_distance

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("nearest_neighbor <word_emb_type> <word_emb>")