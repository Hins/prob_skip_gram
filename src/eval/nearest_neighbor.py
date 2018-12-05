# -*- coding: utf-8 -*-
# @Time        : 2018/11/26 11:22
# @Author      : panxiaotong
# @Description : evaluate by nearest neighbor, qualitative evaluation

import sys
import gensim

from util import extract_features, cosin_distance

if __name__ == "__main__":
    if len(sys.argv) < 15:
        print("nearest_neighbor <word_emb_type> <word_emb> <word_dict> <use_other_info> <word_ids> <pos_ids> "
              "<pos_emb> <parser_ids> <parser_emb> <dict_desc_ids> <kb_ids> <kb_emb> <word_list_file> <output_file>")
        sys.exit()

    word_list = []
    if sys.argv[13].strip() != '':
        with open(sys.argv[13], 'r') as f:
            for line in f:
                word_list.append(line.strip('\r\n'))
            f.close()
    print('word_list len is %d' % len(word_list))
    output_file = open(sys.argv[14], 'w')
    if sys.argv[1].lower() == "file":
        word_emb_dict = {}
        with open(sys.argv[2], 'r') as f:
            for idx, line in enumerate(f):
                word_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        word_dict = {}
        reverse_word_dict = {}
        with open(sys.argv[3], 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split('\t')
                word_dict[elements[0]] = int(elements[1])
                reverse_word_dict[int(elements[1])] = elements[0]
            f.close()
        word_dict_copy = word_dict.copy()

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
                pos_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        parser_dict = {}
        with open(sys.argv[8], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in parser_dict:
                    parser_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                         line.strip('\r\n').split(',')]
            f.close()

        parser_emb_dict = {}
        with open(sys.argv[9], 'r') as f:
            for idx, line in enumerate(f):
                parser_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        dict_desc_dict = {}
        with open(sys.argv[10], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in dict_desc_dict:
                    dict_desc_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                            line.strip('\r\n').split(',')]
            f.close()

        kb_dict = {}
        with open(sys.argv[11], 'r') as f:
            for idx, line in enumerate(f):
                if reverse_word_dict[word_id_list[idx]] not in kb_dict:
                    kb_dict[reverse_word_dict[word_id_list[idx]]] = [int(item) for item in
                                                                     line.strip('\r\n').split(',')]
            f.close()

        kb_emb_dict = {}
        with open(sys.argv[12], 'r') as f:
            for idx, line in enumerate(f):
                # key is word index in dictionary, value is embedding vector
                kb_emb_dict[idx] = [float(item) for item in line.strip('\r\n').split(',')]
            f.close()

        if len(word_list) == 0:
            for k,v in word_dict.items():
                sim_dict = {}
                for kn, vn in word_dict_copy.items():
                    if k == kn:
                        continue
                    sim_vec1 = extract_features(use_other_info, k, word_dict, word_emb_dict, pos_dict,
                                                          pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                                          kb_dict, kb_emb_dict)
                    if sim_vec1 == None:
                        continue
                    sim_vec2 = extract_features(use_other_info, kn, word_dict, word_emb_dict, pos_dict,
                                                          pos_emb_dict, parser_dict, parser_emb_dict, dict_desc_dict,
                                                          kb_dict, kb_emb_dict)
                    if sim_vec2 == None:
                        continue
                    sim_dict[kn] = cosin_distance(sim_vec1, sim_vec2)
                sim_key_list = sorted(sim_dict, key=sim_dict.get, reverse=True)
                rtn_list = []
                for idx, key in enumerate(sim_key_list):
                    if idx > 2:
                        break
                    rtn_list.append(key)
                output_file.write(k + " : " + ",".join(rtn_list) + "\n")
                output_file.flush()
        else:
            for word in word_list:
                sim_dict = {}
                for k,v in word_dict.items():
                    if word == k:
                        continue
                    sim_vec1 = extract_features(use_other_info, word, word_dict, word_emb_dict, pos_dict, pos_emb_dict,
                                                parser_dict, parser_emb_dict, dict_desc_dict, kb_dict, kb_emb_dict)
                    if sim_vec1 == None:
                        continue
                    sim_vec2 = extract_features(use_other_info, k, word_dict, word_emb_dict, pos_dict, pos_emb_dict,
                                                parser_dict, parser_emb_dict, dict_desc_dict, kb_dict, kb_emb_dict)
                    if sim_vec2 == None:
                        continue
                    sim_dict[k] = cosin_distance(sim_vec1, sim_vec2)
                sim_key_list = sorted(sim_dict, key=sim_dict.get, reverse=True)
                rtn_list = []
                for idx, key in enumerate(sim_key_list):
                    if idx > 2:
                        break
                    rtn_list.append(key)
                output_file.write(k + " : " + ",".join(rtn_list) + "\n")
                output_file.flush()
    # just use word vector to get most similarity terms
    elif sys.argv[1].lower() == "w2v":
        model = gensim.models.Word2Vec.load(sys.argv[2])
        word_dict = {}
        with open(sys.argv[3], 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split('\t')
                word_dict[elements[0]] = int(elements[1])
            f.close()
        if len(word_list) == 0:
            for word, idx in word_dict.items():
                if word not in model:
                    continue
                most_top_n_terms = model.wv.most_similar(word, topn=3)
                most_top_n_list = []
                for item in most_top_n_terms:
                    most_top_n_list.append(item[0])
                output_file.write(word + " : " + ",".join(most_top_n_list) + "\n")
        else:
            for word in word_list:
                if word not in model:
                    continue
                most_top_n_terms = model.wv.most_similar(word, topn=3)
                most_top_n_list = [item[0] for item in most_top_n_terms]
                output_file.write(word + " : " + ",".join(most_top_n_list) + "\n")
    elif sys.argv[1].lower() == "glove":
        word_dict = {}
        with open(sys.argv[2], 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split(' ')
                word_dict[elements[0]] = [float(item) for idx, item in enumerate(elements) if idx > 0]
            f.close()

        word_dict_tmp = word_dict.copy()
        if len(word_list) == 0:
            for k,v in word_dict.items():
                sim_dict = {}
                for sub_k, sub_v in word_dict_tmp.items():
                    if k == sub_k:
                        continue
                    sim_dict[sub_k] = cosin_distance(v, sub_v)
                sim_key_list = sorted(sim_dict, key=sim_dict.get, reverse=True)
                rtn_list = []
                for idx, key in enumerate(sim_key_list):
                    if idx > 2:
                        break
                    rtn_list.append(key)
                output_file.write(k + " : " + ",".join(rtn_list) + "\n")
                output_file.flush()
        else:
            for word in word_list:
                sim_dict = {}
                for k,v in word_dict.items():
                    if word == k or word not in word_dict:
                        continue
                    sim_dict[k] = cosin_distance(word_dict[word], v)
                sim_key_list = sorted(sim_dict, key=sim_dict.get, reverse=True)
                rtn_list = []
                for idx, key in enumerate(sim_key_list):
                    if idx > 2:
                        break
                    rtn_list.append(key)
                output_file.write(word + " : " + ",".join(rtn_list) + "\n")
                output_file.flush()
    output_file.close()