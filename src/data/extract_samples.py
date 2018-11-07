# -*- coding: utf-8 -*-
# @Time        : 2018/9/4 18:43
# @Author      : panxiaotong
# @Description : extract samples

import json
import sys
sys.path.append("..")
from util.config import cfg
from stanfordcorenlp import StanfordCoreNLP
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from pyltp import Parser, Postagger
import re
import requests
import string

class GooleKGAPI(object):
    def __init__(self):
        self.api_key = 'AIzaSyAvXCjcZCh7QRMAgcppheJkfUWktGZQg_M'

    def getResult(self,query):
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search?key=' + \
                      self.api_key + '&limit=1&indent=True&query=' + query
        r = requests.get(url=service_url)
        response = json.loads(r.text)
        if "itemListElement" not in response or len(response['itemListElement']) == 0 or \
            "result" not in response['itemListElement'][0] or "@type" not in response['itemListElement'][0]['result']:
            print(query)
            return ""
        return response['itemListElement'][0]['result']['@type']

if __name__ == "__main__":
    if len(sys.argv) < 17:
        print("<extract_samples> <input file> <stanford corenlp library path> <ltp parser model> <ltp pos model> "
              "<word dict output file> <word id output file> <context output file> <part-of-speech output file> "
              "<part-of-speech dict file> <parser output file> <parser dict output file> <dict desc output file> "
              "<kb entity output file> <kb dict output file> <word count dict output file> <bigram cooccurence dict output file>")
        sys.exit()

    nlp = StanfordCoreNLP(sys.argv[2])
    parser = Parser()
    parser.load(sys.argv[3])
    postagger = Postagger()
    postagger.load(sys.argv[4])
    WikiData_url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&language=en&search="
    request_header = {'accept':'application/json'}
    gkg = GooleKGAPI()
    '''
    DBPedia_url = "https://api.dbpedia-spotlight.org/en/annotate?text="
    DBPedia_header = {'accept':'application/json'}
    DBPedia_start_str = "Schema:"
    DBPedia_end_str = ","
    '''
    sub_context_window_size = cfg.context_window_size / 2

    word_dict_output = open(sys.argv[5], 'w')
    word_output = open(sys.argv[6], 'w')
    context_output = open(sys.argv[7], 'w')
    partofspeech_output = open(sys.argv[8], 'w')
    partofspeech_dict_output = open(sys.argv[9], 'w')
    parser_output = open(sys.argv[10], 'w')
    parser_dict_output = open(sys.argv[11], 'w')
    dict_desc_output = open(sys.argv[12], 'w')
    kb_entity_output = open(sys.argv[13], 'w')
    kb_dict_output = open(sys.argv[14], 'w')
    word_count_output = open(sys.argv[15], 'w')
    word_coocur_output = open(sys.argv[16], 'w')

    word_dict = {}
    word_count_dict = {}
    word_coocur_dict = {}
    pos_dict = {}
    parser_dict = {}
    kb_entity_dict = {}
    kb_entity_content_dict = {}
    wordnet_dict = {}
    WikiData_dict = {}
    punctuation_list = string.punctuation
    integ_regex = re.compile(r'-?[1-9]\d*')
    digit_regex = re.compile(r'-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip('\r\n')
            tokenize_list = [str(item) for item in nlp.word_tokenize(line) if str(item) not in punctuation_list and
                             digit_regex.match(str(item)) == None and integ_regex.match(str(item)) == None]
            tokenize_list_len = len(tokenize_list)
            if tokenize_list_len < 1 + cfg.context_window_size:
                continue
            # tuple: [word, pos_tag]
            pos_list = nlp.pos_tag(line)
            postags = postagger.postag(tokenize_list)  # 词性标注
            dependency_parse_list = parser.parse(tokenize_list, postags)  # 句法分析

            word_index = sub_context_window_size
            while word_index < tokenize_list_len - sub_context_window_size:
                cur_word = tokenize_list[word_index]
                if cur_word not in word_dict:
                    word_dict[cur_word] = len(word_dict) + 1
                word_output.write(str(word_dict[cur_word]) + "\n")
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
                cur_pos = str(pos_list[word_index][1])
                if cur_pos not in pos_dict:
                    pos_dict[cur_pos] = len(pos_dict) + 1
                partofspeech_output.write(str(pos_dict[cur_pos]) + "\n")

                parser_list = []
                for i in range(sub_context_window_size + 1):
                    if i == 0:
                        continue
                    if dependency_parse_list[word_index - i].head == word_index:
                        if dependency_parse_list[word_index - i].relation not in parser_dict:
                            parser_dict[dependency_parse_list[word_index - i].relation] = len(parser_dict) + 1
                        parser_list.append(str(parser_dict[dependency_parse_list[word_index - i].relation]))
                    else:
                        parser_list.append("0")
                    if dependency_parse_list[word_index + i].head == word_index:
                        if dependency_parse_list[word_index + i].relation not in parser_dict:
                            parser_dict[dependency_parse_list[word_index + i].relation] = len(parser_dict) + 1
                        parser_list.append(str(parser_dict[dependency_parse_list[word_index + i].relation]))
                    else:
                        parser_list.append("0")
                if len(parser_list) < cfg.context_window_size:
                    print(line)
                parser_output.write(",".join(parser_list) + "\n")

                if cur_word not in wordnet_dict:
                    wordnet_result = wn.synsets(cur_word)
                    if len(wordnet_result) == 0:
                        dict_desc_output.write("0\n")
                        wordnet_dict[cur_word] = ["0"]
                    else:
                        dict_desc_word_list = nlp.word_tokenize(wordnet_result[0].definition())
                        dict_desc_word_id_list = []
                        for word in dict_desc_word_list:
                            if word in punctuation_list:
                                continue
                            if digit_regex.match(word) != None:
                                continue
                            if integ_regex.match(word) != None:
                                continue
                            if word not in word_dict:
                                word_dict[word] = len(word_dict) + 1
                            dict_desc_word_id_list.append(str(word_dict[word]))
                        dict_desc_output.write(",".join(dict_desc_word_id_list) + "\n")
                        wordnet_dict[cur_word] = dict_desc_word_id_list
                else:
                    dict_desc_output.write(",".join(wordnet_dict[cur_word]) + "\n")

                '''
                if cur_word not in kb_entity_content_dict:
                    kb_list = gkg.getResult(cur_word)
                    if isinstance(kb_list, str):
                        kb_entity_output.write("0\n")
                        kb_entity_content_dict[cur_word] = "0"
                        continue
                    kb_list_id = []
                    for entity in kb_list:
                        if entity not in kb_entity_dict:
                            kb_entity_dict[entity] = len(kb_entity_dict) + 1
                        kb_list_id.append(kb_entity_dict[entity])
                    kb_entity_content_dict[cur_word] = ",".join([str(item) for item in kb_list_id])
                    kb_entity_output.write(",".join([str(item) for item in kb_list_id]) + "\n")
                else:
                    kb_entity_output.write(kb_entity_content_dict[cur_word] + "\n")
                '''
                word_index += 1
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