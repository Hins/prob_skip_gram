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
    if len(sys.argv) < 15:
        print("<extract_samples> <input file> <stanford corenlp library path> <word dict output file> <word id output file> "
              "<context output file> <part-of-speech output file> <part-of-speech dict file> <parser output file> "
              "<parser dict output file> <dict desc output file> <kb entity output file> <kb dict output file> "
              "<word count dict output file> <bigram cooccurence dict output file>")
        sys.exit()

    nlp = StanfordCoreNLP(sys.argv[2])
    DBPedia_url = "https://api.dbpedia-spotlight.org/en/annotate?text="
    DBPedia_header = {'accept':'application/json'}
    DBPedia_start_str = "Schema:"
    DBPedia_end_str = ","
    sub_context_window_size = cfg.context_window_size / 2

    word_dict_output = open(sys.argv[3], 'w')
    word_output = open(sys.argv[4], 'w')
    context_output = open(sys.argv[5], 'w')
    partofspeech_output = open(sys.argv[6], 'w')
    partofspeech_dict_output = open(sys.argv[7], 'w')
    parser_output = open(sys.argv[8], 'w')
    parser_dict_output = open(sys.argv[9], 'w')
    dict_desc_output = open(sys.argv[10], 'w')
    kb_entity_output = open(sys.argv[11], 'w')
    kb_dict_output = open(sys.argv[12], 'w')
    word_count_output = open(sys.argv[13], 'w')
    word_coocur_output = open(sys.argv[14], 'w')

    word_dict = {}
    word_count_dict = {}
    word_coocur_dict = {}
    pos_dict = {}
    parser_dict = {}
    kb_entity_dict = {}
    wordnet_dict = {}
    DBPedia_dict = {}
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            tokenize_list = [str(item) for item in nlp.word_tokenize(line)]
            tokenize_list_len = len(tokenize_list)
            if tokenize_list_len < 1 + cfg.context_window_size:
                continue
            # tuple: [word, pos_tag]
            pos_list = nlp.pos_tag(line)
            # tuple: [relation, pointer, pointed] except ROOT relationship
            dependency_parse_list = nlp.dependency_parse(line)

            word_index = sub_context_window_size
            while word_index < tokenize_list_len - sub_context_window_size:
                cur_word = tokenize_list[word_index]
                if cur_word not in word_dict:
                    word_dict[cur_word] = len(word_dict) + 1
                word_output.write(str(word_dict[cur_word]) + "\n")
                word_output.flush()
                if word_dict[cur_word] not in word_count_dict:
                    word_count_dict[word_dict[cur_word]] = 0
                word_count_dict[word_dict[cur_word]] += 1
                for i in range(sub_context_window_size + 1):
                    if i == 0:
                        continue
                    if tokenize_list[word_index - i] not in word_dict:
                        word_dict[tokenize_list[word_index - i]] = len(word_dict) + 1
                    coocur_words_1 = str(word_dict[cur_word]) + cfg.coocur_separator + str(word_dict[tokenize_list[word_index - i]])
                    coocur_words_2 = str(word_dict[tokenize_list[word_index - i]]) + cfg.coocur_separator + str(word_dict[cur_word])
                    if coocur_words_1 in word_coocur_dict:
                        word_coocur_dict[coocur_words_1] += 1
                    elif coocur_words_2 in word_coocur_dict:
                        word_coocur_dict[coocur_words_2] += 1
                    else:
                        word_coocur_dict[coocur_words_1] = 1
                    if tokenize_list[word_index + i] not in word_dict:
                        word_dict[tokenize_list[word_index + i]] = len(word_dict) + 1
                    coocur_words_1 = str(word_dict[cur_word]) + cfg.coocur_separator + str(word_dict[tokenize_list[word_index + i]])
                    coocur_words_2 = str(word_dict[tokenize_list[word_index + i]]) + cfg.coocur_separator + str(word_dict[cur_word])
                    if coocur_words_1 in word_coocur_dict:
                        word_coocur_dict[coocur_words_1] += 1
                    elif coocur_words_2 in word_coocur_dict:
                        word_coocur_dict[coocur_words_2] += 1
                    else:
                        word_coocur_dict[coocur_words_1] = 1
                context_list = []
                for i in range(cfg.context_window_size + 1):
                    if word_index - sub_context_window_size + i == word_index:
                        continue
                    context_list.append(str(word_dict[tokenize_list[word_index - sub_context_window_size + i]]))
                context_output.write(",".join(context_list) + "\n")
                context_output.flush()
                cur_pos = str(pos_list[word_index][1])
                if cur_pos not in pos_dict:
                    pos_dict[cur_pos] = len(pos_dict) + 1
                partofspeech_output.write(str(pos_dict[cur_pos]) + "\n")
                partofspeech_output.flush()

                parser_list = []
                for i in range(cfg.context_window_size + 1):
                    parser_index = word_index - sub_context_window_size + i    # context word index
                    if parser_index == word_index:
                        continue
                    for item in dependency_parse_list:
                        if str(item[0]) not in parser_dict:
                            parser_dict[str(item[0])] = len(parser_dict) + 1
                        if str(item[0]) != "ROOT":
                            if item[2] != parser_index + 1:
                                continue
                            if item[1] == word_index + 1:
                                parser_list.append(str(parser_dict[str(item[0])]))
                            else:
                                parser_list.append("0")
                            break
                        else:
                            if item[2] != parser_index + 1:
                                continue
                            parser_list.append("0")
                            break
                parser_output.write(",".join(parser_list) + "\n")
                parser_output.flush()

                if cur_word not in wordnet_dict:
                    wordnet_result = wn.synsets(cur_word)
                    if len(wordnet_result) == 0:
                        dict_desc_output.write("0\n")
                        wordnet_dict[cur_word] = ["0"]
                    else:
                        dict_desc_word_list = nlp.word_tokenize(wordnet_result[0].definition())
                        dict_desc_word_id_list = []
                        for word in dict_desc_word_list:
                            if word not in word_dict:
                                word_dict[word] = len(word_dict) + 1
                            dict_desc_word_id_list.append(str(word_dict[word]))
                        dict_desc_output.write(",".join(dict_desc_word_id_list) + "\n")
                        wordnet_dict[cur_word] = dict_desc_word_id_list
                else:
                    dict_desc_output.write(",".join(wordnet_dict[cur_word]) + "\n")
                dict_desc_output.flush()

                if cur_word not in DBPedia_dict:
                    r = requests.get(DBPedia_url + cur_word, headers=DBPedia_header)
                    if r.status_code != 200:
                        kb_entity_output.write("0\n")
                    else:
                        try:
                            json_obj = json.loads(str(r.text))
                        except Exception,err:
                            #print(r.text)
                            print("json.loads() failed")
                            kb_entity_output.write("0\n")
                            word_index += 1
                            continue
                        if "Resources" not in json_obj:
                            kb_entity_output.write("0\n")
                            DBPedia_dict[cur_word] = "0"
                        else:
                            DB_str = str(json_obj["Resources"][0]["@types"])
                            start_offset = DB_str.find(DBPedia_start_str)
                            if start_offset == -1:
                                kb_entity_output.write("0\n")
                                DBPedia_dict[cur_word] = "0"
                            else:
                                end_offset = DB_str.find(DBPedia_end_str, start_offset + len(DBPedia_start_str))
                                if end_offset == -1:
                                    kb_entity_output.write("0\n")
                                    DBPedia_dict[cur_word] = "0"
                                else:
                                    entity = DB_str[start_offset + len(DBPedia_start_str):end_offset]
                                    if entity not in kb_entity_dict:
                                        kb_entity_dict[entity] = len(kb_entity_dict) + 1
                                    kb_entity_output.write(str(kb_entity_dict[entity]) + "\n")
                                    DBPedia_dict[cur_word] = str(kb_entity_dict[entity])
                else:
                    kb_entity_output.write(DBPedia_dict[cur_word] + "\n")
                kb_entity_output.flush()
                word_index += 1
            print(line)
        f.close()

    word_output.close()
    context_output.close()
    partofspeech_output.close()
    parser_output.close()
    dict_desc_output.close()
    kb_entity_output.close()

    for k,v in word_dict.items():
        word_dict_output.write(k + "\t" + str(v) + "\n")
    for k,v in pos_dict.items():
        partofspeech_dict_output.write(k + "\t" + str(v) + "\n")
    for k,v in parser_dict.items():
        parser_dict_output.write(k + "\t" + str(v) + "\n")
    for k,v in kb_entity_dict.items():
        kb_dict_output.write(k + "\t" + str(v) + "\n")
    for k,v in word_count_dict.items():
        word_count_output.write(str(k) + "\t" + str(v) + "\n")
    for k,v in word_coocur_dict.items():
        word_coocur_output.write(k + "\t" + str(v) + "\n")

    nlp.close()
    word_dict_output.close()
    partofspeech_dict_output.close()
    parser_dict_output.close()
    kb_dict_output.close()
    word_count_output.close()
    word_coocur_output.close()