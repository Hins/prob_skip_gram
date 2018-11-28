# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 18:24
# @Author      : panxiaotong
# @Description : extract movie review sentiment data from mc dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("extract_from_movie_review <input_file> <word_dict> <output_file>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()
    output_file = open(sys.argv[3], 'w')
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('\t')
            words = [item for item in elements[2] if item.strip() != ""]
            sent_ids = []
            for word in words:
                sent_ids.append("0" if word not in word_dict else str(word_dict[word]))
            if sent_ids.count("0") != len(sent_ids):
                output_file.write(",".join(sent_ids) + "\t" + elements[3] + "\n")
        f.close()
    output_file.close()