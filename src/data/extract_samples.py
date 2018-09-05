# -*- coding: utf-8 -*-
# @Time        : 2018/9/4 18:43
# @Author      : panxiaotong
# @Description : extract samples

import sys
sys.path.append("..")
from util.config import cfg
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import wordnet as wn
import requests

if __name__ == "__main__":
    if len(sys.argv) < 10:
        print("<extract_word_id> <input file> <stanford corenlp library path> <word dict output file>"
              "<word id output file> <context output file> <part-of-speech output file> <parser output file>"
              "<dict desc output file> <kb entity output file>")
        sys.exit()

    nlp = StanfordCoreNLP(sys.argv[2])
    DBPedia_url = "https://api.dbpedia-spotlight.org/en/annotate?text="
    DBPedia_header = {'accept':'application/json'}
    sub_context_window_size = cfg.context_window_size / 2

    word_dict_output = open(sys.argv[3], 'w')
    word_output = open(sys.argv[4], 'w')
    context_output = open(sys.argv[5], 'w')
    partofspeech_output = open(sys.argv[6], 'w')
    parser_output = open(sys.argv[7], 'w')
    dict_desc_output = open(sys.argv[8], 'w')
    kb_entity_output = open(sys.argv[9], 'w')

    word_dict = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            tokenize_list = nlp.word_tokenize(line)

            for word in tokenize_list:
                word = str(word)
                if word not in word_dict:
                    word_dict[word] = len(word_dict) + 1

                print(wn.synsets(str(item))[0].definition())
                r = requests.get(DBPedia_url + str(item), headers=DBPedia_header)
                print(r.status_code)
                print(str(r.text))
                break
            pos_list = nlp.pos_tag(line)
            for item in pos_list:
                print(str(item[0]))
                print(str(item[1]))
                break
            dependency_parse_list = nlp.dependency_parse(line)
            for item in dependency_parse_list:
                print(str(item[0]))
                print(str(item[1]))
                print(str(item[2]))
                break
        f.close()

    word_dict_output.close()
    word_output.close()
    context_output.close()
    partofspeech_output.close()
    parser_output.close()
    dict_desc_output.close()
    kb_entity_output.close()