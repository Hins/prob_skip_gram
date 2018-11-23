# -*- coding: utf-8 -*-
# @Time        : 2018/11/23 10:45
# @Author      : panxiaotong
# @Description : extract entailment samples from SICK dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("extract_entailment_samples <input file> <word dict> <output file1> <output file2>")
        sys.exit()

    word_dict = {}
    with open(sys.argv[2], 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_dict[elements[0]] = elements[1]
        f.close()

    output_single = open(sys.argv[3], 'w')
    output_complex = open(sys.argv[4], 'w')
    single_class_dict = {}
    complex_class_dict = {}
    with open(sys.argv[1], 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            elements = line.strip('\r\n').split('\t')
            sent1 = [item for item in elements[1].split(' ') if item.strip() != ""]
            id1_list = []
            for word in sent1:
                if word in word_dict:
                    id1_list.append(word_dict[word])
                else:
                    id1_list.append("0")
            flag = False
            for id in id1_list:
                if id != "0":
                    flag = True
                    break
            if flag == False:
                continue
            sent2 = [item for item in elements[2].split(' ') if item.strip() != ""]
            id2_list = []
            for word in sent2:
                if word in word_dict:
                    id2_list.append(word_dict[word])
                else:
                    id2_list.append("0")
            flag = False
            for id in id2_list:
                if id != "0":
                    flag = True
                    break
            if flag == False:
                continue
            class_single = elements[3]
            if class_single not in single_class_dict:
                single_class_dict[class_single] = len(single_class_dict)
            score = elements[4]
            output_single.write(",".join(id1_list) + "\t" + ",".join(id2_list) +
                                "\t" + score + "\t" + str(single_class_dict[class_single]) + "\n")
            sub_class1_single = elements[5]
            sub_class2_single = elements[6]
            total_class = sub_class1_single + "_" + sub_class2_single
            if total_class not in complex_class_dict:
                complex_class_dict[total_class] = len(complex_class_dict)
            output_complex.write(",".join(id1_list) + "\t" + ",".join(id2_list) +
                                 "\t" + score + "\t" + str(complex_class_dict[total_class]) + "\n")
        f.close()

    output_single.close()
    output_complex.close()