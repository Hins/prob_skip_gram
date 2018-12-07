# -*- coding: utf-8 -*-
# @Time        : 2018/11/21 14:32
# @Author      : panxiaotong
# @Description : extract negative samples

import random
import sys
sys.path.append("..")
from config.config import cfg
import numpy as np

import resource
rsrc=resource.RLIMIT_AS
res_mem="15G"
memlimit=float(res_mem[:-1]) * 1073741824
resource.setrlimit(rsrc, (memlimit, memlimit))
soft, hard = resource.getrlimit(rsrc)
print("memory limit as:", soft, hard)

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("extract_negative_samples <target id> <context> <word dict> <output file> <index> <split_size>")
        sys.exit()

    target_list = []
    sample_size = 0
    with open(sys.argv[1], 'r') as f:
        for line in f:
            target_list.append(int(line.strip('\r\n')))
            sample_size += 1
        f.close()

    word_dictionary_size = len(open(sys.argv[3]).readlines()) + 1

    index_begin = (sample_size / int(sys.argv[6])) * int(sys.argv[5])
    index_end = (sample_size / int(sys.argv[6])) * (1 + int(sys.argv[5]))
    if int(sys.argv[5]) == int(sys.argv[6]) - 1:
        index_end = sample_size

    output_file = open(sys.argv[4], 'w')
    context_list = []
    with open(sys.argv[2], 'r') as f:
        for idx, line in enumerate(f):
            if idx < index_begin:
                continue
            if idx >= index_end:
                break
            words = line.strip('\r\n').split(',')
            sub_labels = np.full(shape=[cfg.negative_sample_size],
                                 fill_value=word_dictionary_size + 1, dtype=np.int32)
            iter = 0
            while iter < cfg.negative_sample_size:
                r = random.randint(0, word_dictionary_size - 1)
                if r == target_list[idx]:
                    continue
                flag = False
                for word in sub_labels:
                    if r == word:
                        flag = True
                        break
                if flag == True:
                    continue
                for word in words:
                    if r == word:
                        flag = True
                        break
                if flag == False:
                    sub_labels[iter] = r
                    iter += 1
            np.random.shuffle(sub_labels)  # shuffle positive and negative samples
            output_file.write(','.join([str(item) for item in sub_labels.tolist()]) + '\n')
            output_file.flush()

            context_list.append(line.strip('\r\n').split(','))
        f.close()