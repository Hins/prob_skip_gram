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

if __name__ == "__main__":
    if len(sys.argv) < 10:
        print("<extract_samples> <input file> <stanford corenlp library path> <word dict output file>"
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
    pos_dict = {}
    parser_dict = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            tokenize_list = [str(item) for item in nlp.word_tokenize(line)]
            tokenize_list_len = len(tokenize_list)
            if tokenize_list_len < 1 + cfg.context_window_size:
                continue
            pos_list = nlp.pos_tag(line)
            dependency_parse_list = nlp.dependency_parse(line)

            word_index = sub_context_window_size
            while word_index < tokenize_list_len - sub_context_window_size:
                cur_word = tokenize_list[word_index]
                if cur_word not in word_dict:
                    word_dict[cur_word] = len(word_dict) + 1
                word_output.write(str(word_dict[cur_word]) + "\n")
                for i in range(sub_context_window_size + 1):
                    if i == 0:
                        continue
                    if tokenize_list[word_index - i] not in word_dict:
                        word_dict[tokenize_list[word_index - i]] = len(word_dict) + 1
                    if tokenize_list[word_index + i] not in word_dict:
                        word_dict[tokenize_list[word_index + i]] = len(word_dict) + 1
                context_list = []
                for i in range(cfg.context_window_size):
                    if word_index - sub_context_window_size + i == word_index:
                        continue
                    context_list.append(str(word_dict[tokenize_list[word_index - sub_context_window_size + i]]))
                context_output.write(",".join(context_list) + "\n")
                cur_pos = str(pos_list[word_index][1])
                if cur_pos not in pos_dict:
                    pos_dict[cur_pos] = len(pos_dict) + 1
                partofspeech_output.write(str(pos_dict[cur_pos]) + "\n")

                parser_list = []
                for i in range(cfg.context_window_size):
                    parser_index = word_index - sub_context_window_size + i
                    for item in dependency_parse_list:
                        if item[2] != parser_index:
                            continue
                        if item[1] == word_index:
                            if str(item[0]) not in parser_dict:
                                parser_dict[str(item[0])] = len(parser_dict) + 1
                            parser_list.append(str(parser_dict[str(item[0])]))
                        else:
                            parser_list.append("0")
                parser_output.write(",".join(parser_list) + "\n")

                wordnet_result = wn.synsets(cur_word)
                if wordnet_result is None:
                    dict_desc_output.write("\n")
                else:
                    dict_desc_word_list = nlp.word_tokenize(wn.synsets(cur_word)[0].definition())
                    dict_desc_word_id_list = []
                    for word in dict_desc_word_list:
                        if word not in word_dict:
                            word_dict[word] = len(word_dict) + 1
                        dict_desc_word_id_list.append(str(word_dict[word]))
                    dict_desc_output.write(",".join(dict_desc_word_id_list) + "\n")

                r = requests.get(DBPedia_url + cur_word, headers=DBPedia_header)
                json_obj = json.loads(str(r.text))
                if "Resources" not in json_obj:
                    kb_entity_output.write("0\n")
                else:
                    print(type(json_obj["Resources"]["types"]))

                word_index += 1
                break
            break
        f.close()

    word_dict_output.close()
    word_output.close()
    context_output.close()
    partofspeech_output.close()
    parser_output.close()
    dict_desc_output.close()
    kb_entity_output.close()