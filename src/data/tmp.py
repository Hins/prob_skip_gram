# -*- coding: utf-8 -*-
# @Time        : 2018/9/4 18:43
# @Author      : panxiaotong
# @Description : extract samples

import json
import sys
sys.path.append("..")
from util.config import cfg
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import wordnet as wn
import requests
from pyltp import Parser, Postagger

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("<extract_samples> <input file> <stanford nlp path> <ltp parser> <ltp pos> <parser output file> <parser dict output file>")
        sys.exit()

    nlp = StanfordCoreNLP(sys.argv[2])
    sub_context_window_size = cfg.context_window_size / 2

    parser_output = open(sys.argv[5], 'w')
    parser_dict_output = open(sys.argv[6], 'w')
    parser_dict = {}

    parser = Parser()
    print("start parser")
    parser.load(sys.argv[3])
    print("load parser complete")
    postagger = Postagger()
    print("start lpos")
    postagger.load(sys.argv[4])
    print("load pos complete")
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            tokenize_list = [str(item) for item in nlp.word_tokenize(line)]
            tokenize_list_len = len(tokenize_list)
            if tokenize_list_len < 1 + cfg.context_window_size:
                continue
            print("process pos")
            postags = postagger.postag(tokenize_list)  # 词性标注
            print("process parser")
            dependency_parse_list = parser.parse(tokenize_list, postags)  # 句法分析

            word_index = sub_context_window_size
            while word_index < tokenize_list_len - sub_context_window_size:
                cur_word = tokenize_list[word_index]
                parser_list = []
                print(line)
                for i in range(cfg.context_window_size + 1):
                    for i, arc in enumerate(dependency_parse_list):
                        print(i)
                        print(arc)
                break
                '''
                if len(parser_list) < cfg.context_window_size:
                    print(line)
                    print(cur_word)
                '''
                parser_output.write(",".join(parser_list) + "\n")
                parser_output.flush()

                word_index += 1
            break
        f.close()

    for k,v in parser_dict.items():
        parser_dict_output.write(k + "\t" + str(v) + "\n")

    nlp.close()
    parser_dict_output.close()