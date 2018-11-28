# -*- coding: utf-8 -*-
# @Time        : 2018/11/28 15:21
# @Author      : panxiaotong
# @Description : word analogy evaluation

import sys
import numpy as np
import gensim

from util import extract_features, cosin_distance

if __name__ == "__main__":
    if len(sys.argv) < 14:
        print("word_analogy <input file> <word_emb_type> <word_emb> <word_dict> <use_other_info> <word_ids> "
              "<pos_ids> <pos_emb> <parser_ids> <parser_emb> <dict_desc_ids> <kb_ids> <kb_emb>")
        sys.exit()

    analogy_list = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            analogy_list.append(line.strip('\r\n').split('\t'))
        f.close()

    if sys.argv[2].lower() == "file":
        word_emb_dict = {}
        with open(sys.argv[3], 'r') as f:
            for idx, line in enumerate(f):
                word_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        word_dict = {}
        reverse_word_dict = {}
        with open(sys.argv[4], 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split('\t')
                word_dict[elements[0]] = int(elements[1])
                reverse_word_dict[int(elements[1])] = elements[0]
            f.close()
        word_dict_copy = word_dict.copy()

        use_other_info = sys.argv[5].lower()

        word_id_list = []
        with open(sys.argv[6], 'r') as f:
            for line in f:
                word_id_list.append(int(line.strip('\r\n')))
            f.close()

        pos_dict = {}
        with open(sys.argv[7], 'r') as f:
            for idx, line in enumerate(f):
                element = int(line.strip('\r\n'))
                if reverse_word_dict[word_id_list[idx]] not in pos_dict:
                    pos_dict[reverse_word_dict[word_id_list[idx]]] = element
            f.close()

        pos_emb_dict = {}
        with open(sys.argv[8], 'r') as f:
            for idx, line in enumerate(f):
                pos_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        parser_dict = {}
        with open(sys.argv[9], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in parser_dict:
                    parser_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                         line.strip('\r\n').split(',')]
            f.close()

        parser_emb_dict = {}
        with open(sys.argv[10], 'r') as f:
            for idx, line in enumerate(f):
                parser_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        dict_desc_dict = {}
        with open(sys.argv[11], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in dict_desc_dict:
                    dict_desc_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                            line.strip('\r\n').split(',')]
            f.close()

        kb_dict = {}
        with open(sys.argv[12], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in kb_dict:
                    kb_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                     line.strip('\r\n').split(',')]
            f.close()

        kb_emb_dict = {}
        with open(sys.argv[13], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                kb_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        new_feature_dict = {}
        for k,v in word_dict.items():
            new_feature_dict[k] = extract_features(use_other_info, k, word_dict, word_emb_dict, pos_dict,
                                      pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                      kb_dict, kb_emb_dict)
            if new_feature_dict[k] is None:
                print("%s feature extraction failed" % k)

        counter = 0
        real_counter = 0
        top_1 = 0
        top_3 = 0
        for com_item in analogy_list:
            counter += 1
            flag = False
            for element in com_item:
                if element not in word_dict:
                    flag = True
                    break
            if flag is True:
                continue
            a_vec = extract_features(use_other_info, com_item[0], word_dict, word_emb_dict, pos_dict,
                                      pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                      kb_dict, kb_emb_dict)
            if a_vec is None:
                continue
            b_vec = extract_features(use_other_info, com_item[1], word_dict, word_emb_dict, pos_dict,
                                      pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                      kb_dict, kb_emb_dict)
            if b_vec is None:
                continue
            c_vec = extract_features(use_other_info, com_item[2], word_dict, word_emb_dict, pos_dict,
                                      pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                      kb_dict, kb_emb_dict)
            if c_vec is None:
                continue
            d_vec = extract_features(use_other_info, com_item[3], word_dict, word_emb_dict, pos_dict,
                                     pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                     kb_dict, kb_emb_dict)
            if d_vec is None:
                continue
            real_counter += 1
            new_vec = np.subtract(np.asarray(c_vec, dtype=np.float32),
                np.subtract(np.asarray(a_vec, dtype=np.float32), np.asarray(b_vec, dtype=np.float32)))
            score_dict = {}
            for k,v in new_feature_dict.items():
                score_dict[k] = cosin_distance(new_vec.tolist(), v)
            sim_key_list = sorted(score_dict, key=score_dict.get, reverse=True)
            for idx, key in enumerate(sim_key_list):
                if idx == 0 and key == com_item[3]:
                    top_1 += 1
                    break
                if key == com_item[3]:
                    top_3 += 1
                if idx > 2:
                    break
        if real_counter > 0:
            top_1 = float(top_1) / float(real_counter)
            top_3 = float(top_3) / float(top_3)
        else:
            top_1 = 0.0
            top_3 = 0.0
        print("counter is %d, real_counter is %d, top_1 precision is %f, top_3 precision is %f" %
              (counter, real_counter, float("{0:.3f}".format(top_1)), float("{0:.3f}".format(top_3))))
    elif sys.argv[2].lower() == "w2v":
        model = gensim.models.Word2Vec.load(sys.argv[3])
        counter = 0
        real_counter = 0
        top_1 = 0
        top_3 = 0
        for com_item in analogy_list:
            counter += 1
            if com_item[0] not in model or \
                com_item[1] not in model or \
                com_item[2] not in model or \
                com_item[3] not in model:
                continue
            real_counter += 1
            new_vec = np.subtract(np.asarray(model[com_item[2]], dtype=np.float32),
                                  np.subtract(np.asarray(model[com_item[0]], dtype=np.float32), dtype=np.float32),
                                              np.asarray(model[com_item[1]], dtype=np.float32))
            score_dict = {}
            for k,v in model.wv.vocab.item():
                score_dict[k] = cosin_distance(new_vec.tolist(), model[k])
            sim_key_list = sorted(score_dict, key=score_dict.get, reverse=True)
            for idx, key in enumerate(sim_key_list):
                if idx == 0 and key == com_item[3]:
                    top_1 += 1
                    break
                if key == com_item[3]:
                    top_3 += 1
                if idx > 2:
                    break
        if real_counter > 0:
            top_1 = float(top_1) / float(real_counter)
            top_3 = float(top_3) / float(top_3)
        else:
            top_1 = 0.0
            top_3 = 0.0
        print("counter is %d, real_counter is %d, top_1 precision is %f, top_3 precision is %f" %
              (counter, real_counter, float("{0:.3f}".format(top_1)), float("{0:.3f}".format(top_3))))