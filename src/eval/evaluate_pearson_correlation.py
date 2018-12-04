# -*- coding: utf-8 -*-
# @Time        : 2018/11/13 10:45
# @Author      : panxiaotong
# @Description : evaluate by pearson correlation

import numpy as np
import operator
import sys
import gensim
import scipy.stats as stats
from util import extract_features, cosin_distance

if __name__ == "__main__":
    """
    Args:
        word_emb_type: word embed training method, file/w2v, latter was got from Gensim
        word_emb: word embedding file
        word_dict: word dictionary file
        use_other_info: whether to use other information, false or word,pos,parser,dict,kb
            split by ,
        word_ ids: target word id file
        pos_ids: part-of-speech id file
        pos_emb: part-of-speech embedding file
        parser_ids: parser id file
        parser_emb: parser embedding file
        dict_desc_ids: dictionary description ids file
        kb_ids: knowledge-graph entity ids file
        kb_emb: knowledge-graph embedding file
        evaluation_ds: evaluation dataset file
    """
    if len(sys.argv) < 14:
        print("evaluate_pearson_correlation <word_emb_type> <word_emb> <word_dict> <use_other_info> <word_ids> <pos_ids> "
              "<pos_emb> <parser_ids> <parser_emb> <dict_desc_ids> <kb_ids> <kb_emb> <evaluation_ds>")
        sys.exit()

    if sys.argv[1].lower() == "file":
        word_emb_dict = {}
        with open(sys.argv[2], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                word_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        word_dict = {}
        reverse_word_dict = {}
        with open(sys.argv[3], 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split('\t')
                # key is word, value is word index in dictionary
                word_dict[elements[0]] = int(elements[1])
                # key is word index in dictionary, value is word
                reverse_word_dict[int(elements[1])] = elements[0]
            f.close()

        use_other_info = sys.argv[4].lower()

        word_id_list = []
        with open(sys.argv[5], 'r') as f:
            for line in f:
                word_id_list.append(int(line.strip('\r\n')))
            f.close()

        pos_dict = {}
        with open(sys.argv[6], 'r') as f:
            for idx, line in enumerate(f):
                element = int(line.strip('\r\n'))
                if reverse_word_dict[word_id_list[idx]] not in pos_dict:
                    pos_dict[reverse_word_dict[word_id_list[idx]]] = element
            f.close()

        pos_emb_dict = {}
        with open(sys.argv[7], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                pos_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        parser_dict = {}
        with open(sys.argv[8], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in parser_dict:
                    parser_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in line.strip('\r\n').split(',')]
            f.close()

        parser_emb_dict = {}
        with open(sys.argv[9], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                parser_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        dict_desc_dict = {}
        with open(sys.argv[10], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in dict_desc_dict:
                    dict_desc_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in line.strip('\r\n').split(',')]
            f.close()

        kb_dict = {}
        with open(sys.argv[11], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in kb_dict:
                    kb_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in line.strip('\r\n').split(',')]
            f.close()

        kb_emb_dict = {}
        with open(sys.argv[12], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                kb_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        WORD_SEPARATOR = '_'
        index_dict = {}
        pred_score_dict = {}
        with open(sys.argv[13], 'r') as f:
            counter = 0
            real_counter = 0
            for idx, line in enumerate(f):
                real_counter += 1
                elements = line.strip('\r\n').split('\t')
                word_1 = elements[0]
                word_2 = elements[1]
                score = float(elements[2])
                if use_other_info == 'false':
                    if word_1 not in word_dict or word_2 not in word_dict:
                        continue
                    pred_score_dict[counter] = cosin_distance(word_emb_dict[word_dict[word_1]],
                                                              word_emb_dict[word_dict[word_2]])
                    index_dict[counter] = score
                    counter += 1
                else:
                    sim_vec1 = extract_features(use_other_info, word_1, word_dict, word_emb_dict,
                                                            pos_dict, pos_emb_dict, parser_dict, parser_emb_dict,
                                                            dict_desc_dict, kb_dict, kb_emb_dict)
                    if sim_vec1 == None:
                        continue
                    sim_vec2 = extract_features(use_other_info, word_2, word_dict, word_emb_dict,
                                                pos_dict, pos_emb_dict, parser_dict, parser_emb_dict,
                                                dict_desc_dict, kb_dict, kb_emb_dict)
                    if sim_vec2 == None:
                        continue
                    pred_score_dict[counter] = cosin_distance(sim_vec1, sim_vec2)
                    index_dict[counter] = score
                    counter += 1
            f.close()
        print("counter is %d, real_counter is %d" % (counter, real_counter))
        sorted_x = sorted(index_dict.items(), key=operator.itemgetter(1))
        real_value_list = []
        for k,v in sorted_x:
            real_value_list.append(k)
        sorted_x = sorted(pred_score_dict.items(), key=operator.itemgetter(1))
        pred_value_list = []
        for k,v in sorted_x:
            pred_value_list.append(k)
        print("{0:.6f}".format(stats.pearsonr(real_value_list, pred_value_list)[0]))
    # Gensim word2vec function, not support other information
    elif sys.argv[1].lower() == "w2v":
        model = gensim.models.Word2Vec.load(sys.argv[2])
        index_dict = {}
        pred_score_dict = {}
        with open(sys.argv[13], 'r') as f:
            counter = 0
            real_counter = 0
            for idx, line in enumerate(f):
                real_counter += 1
                elements = line.strip('\r\n').split('\t')
                word_1 = elements[0]
                word_2 = elements[1]
                score = float(elements[2])
                if word_1 not in model or word_2 not in model:
                    continue
                pred_score_dict[counter] = cosin_distance(model[word_1],
                                                          model[word_2])
                index_dict[counter] = score
                counter += 1
            f.close()
        print("counter is %d, real_counter is %d" % (counter, real_counter))
        sorted_x = sorted(index_dict.items(), key=operator.itemgetter(1))
        real_value_list = []
        for k,v in sorted_x:
            real_value_list.append(k)
        sorted_x = sorted(pred_score_dict.items(), key=operator.itemgetter(1))
        pred_value_list = []
        for k,v in sorted_x:
            pred_value_list.append(k)
        print("{0:.6f}".format(stats.pearsonr(real_value_list, pred_value_list)[0]))
    elif sys.argv[1].lower() == "glove":
        word_dict = {}
        with open(sys.argv[2], 'r') as f:
            for idx, line in enumerate(f):
                elements = line.strip('\r\n').split(' ')
                word_dict[elements[0]] = [float(item) for idx, item in enumerate(elements) if idx > 0]
            f.close()
        print('word_dict len is %d' % len(word_dict))
        index_dict = {}
        pred_score_dict = {}
        with open(sys.argv[13], 'r') as f:
            counter = 0
            real_counter = 0
            for idx, line in enumerate(f):
                real_counter += 1
                elements = line.strip('\r\n').split('\t')
                word_1 = elements[0]
                word_2 = elements[1]
                score = float(elements[2])
                if word_1 not in word_dict or word_2 not in word_dict:
                    continue
                pred_score_dict[counter] = cosin_distance(word_dict[word_1],
                                                          word_dict[word_2])
                index_dict[counter] = score
                counter += 1
            f.close()
        print("counter is %d, real_counter is %d" % (counter, real_counter))
        sorted_x = sorted(index_dict.items(), key=operator.itemgetter(1))
        real_value_list = []
        for k,v in sorted_x:
            real_value_list.append(k)
        sorted_x = sorted(pred_score_dict.items(), key=operator.itemgetter(1))
        pred_value_list = []
        for k,v in sorted_x:
            pred_value_list.append(k)
        print("{0:.6f}".format(stats.pearsonr(real_value_list, pred_value_list)[0]))