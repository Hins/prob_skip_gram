# -*- coding: utf-8 -*-
# @Time        : 2018/9/3 18:24
# @Author      : panxiaotong
# @Description : probabilistic skip-gram

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tensorflow.contrib import rnn
import tensorflow as tf
from util.config import cfg
import numpy as np

word_dictionary_size = 0
parser_dictionary_size = 0
partofspeech_dictionary_size = 0
kb_relation_dictionary_size = 0

target = []
context = []
context_prob = []
pos = []
parser = []
dict_desc = []
kb_entity = []
def load_sample(target_file, word_dict_file, context_file, pos_file, pos_dict_file,
                parser_file, parser_dict_file, dict_desc_file, kb_entity_file,
                kb_entity_dict_file, word_count_file, word_coocur_file):
    global target, context, context_prob, pos, parser, dict_desc, kb_entity
    word_count_dict = {}
    word_coocur_dict = {}
    with open(word_count_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_count_dict[elements[0]] = int(elements[1])
        f.close()
    with open(word_coocur_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_coocur_dict[elements[0]] = int(elements[1])
        f.close()
    with open(target_file, 'r') as f:
        for line in f:
            target.append(line.strip('\r\n'))
        f.close()
    word_dictionary_size = len(open(word_dict_file).readlines()) + 1
    with open(context_file, 'r') as f:
        for index, line in enumerate(f):
            context_word_ids = line.strip('\r\n').split(',')
            context.append(context_word_ids)
            sub_context_prob = []
            if target[index] not in word_count_dict:
                for i in range(len(context_word_ids)):
                    sub_context_prob.append(1.0 / float(cfg.context_window_size))
            else:
                x_i = word_count_dict[target[index]]
                accumulate_count = 0
                for co_word in context_word_ids:
                    co_name1 = target[index] + cfg.coocur_separator + co_word
                    co_name2 = co_word + cfg.coocur_separator + target[index]
                    if co_name1 in word_coocur_dict:
                        sub_context_prob.append(word_coocur_dict[co_name1])
                        accumulate_count += word_coocur_dict[co_name1]
                    elif co_name2 in word_coocur_dict:
                        sub_context_prob.append(word_coocur_dict[co_name2])
                        accumulate_count += word_coocur_dict[co_name2]
                    else:
                        sub_context_prob.append(0)
                if accumulate_count == 0:
                    sub_context_prob = []
                    for i in range(len(context_word_ids)):
                        sub_context_prob.append(1.0 / float(cfg.context_window_size))
                else:
                    sub_context_prob = [float(item) / float(accumulate_count) for item in sub_context_prob]
            context_prob.append(sub_context_prob)
        f.close()
    with open(pos_file, 'r') as f:
        for line in f:
            pos.append(line.strip('\r\n'))
        f.close()
    partofspeech_dictionary_size = len(open(pos_dict_file).readlines()) + 1
    with open(parser_file, 'r') as f:
        for line in f:
            parser_list = line.strip('\r\n').replace(' ','').split(',')
            if len(parser_list) < cfg.context_window_size:
                supplement_parser_size = cfg.context_window_size - len(parser_list)
                for i in range(supplement_parser_size):
                    parser_list.append(0)
            parser.append(parser_list)
        f.close()
    parser_dictionary_size = len(open(parser_dict_file).readlines()) + 1
    with open(dict_desc_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(',')
            desc_len = len(elements)
            dict_desc_list = []
            if desc_len >= cfg.dict_time_step:
                for index, item in enumerate(elements):
                    if index >= cfg.dict_time_step:
                        break
                    dict_desc_list.append(item)
            else:
                dict_desc_list = elements[:]
                remain_len = cfg.dict_time_step - desc_len
                for i in range(remain_len):
                    dict_desc_list.append(0)
            dict_desc.append(dict_desc_list)
        f.close()
    with open(kb_entity_file, 'r') as f:
        for line in f:
            kb_entity_list = line.strip('\r\n').split(',')
            if len(kb_entity_list) > cfg.kb_relation_length:
                kb_entity_list = kb_entity_list[0:cfg.kb_relation_length]
            else:
                supplement_kb_size = cfg.kb_relation_length - len(kb_entity_list)
                for i in range(supplement_kb_size):
                    kb_entity_list.append("0")
            kb_entity.append(kb_entity_list)
        f.close()
    kb_relation_dictionary_size = len(open(kb_entity_dict_file).readlines()) + 1

    target = np.asarray(target)
    context = np.asarray(context)
    context_prob = np.asarray(context_prob)
    pos = np.asarray(pos)
    parser = np.asarray(parser)
    dict_desc = np.asarray(dict_desc)
    kb_entity = np.asarray(kb_entity)

    return [word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size, kb_relation_dictionary_size]

class PSGModel():
    def __init__(self, sess):
        """
        :param sess: session passed by main entry function
        :algorithm
        1. define W_{word_id},\ W_{pos},\ W_{parser},\ W_{kb} embedding learnable variables
        2. aggregate all parser information of one word by vector addition, s_{parser}
        3. calculate dictionary information by lstm, use last state as dictionary state, s_{dict}
        4. aggretate all knowledge-base entities of one word by vector addition, s_{kb}
        5. define 5 U variables in attention, U_{word_id}^{att},\ U_{pos}^{att},\ U_{parser}^{att},\ U_{dict}^{att},\ U_{kb}^{att}
        6. extract embedding info out of target layer, leverage lstm to calculate out hidden state notated as h_{0},\ h_{1},\ h_{2},\ h_{3}
        7. e_{i\^word\_id} = v^{T} \times tanh(W_{att} \times h_{i} + U_{word_id}^{att} \times s_{word_id})
        8. e_{i\^pos} = v^{T} \times tanh(W_{att} \times h_{i} + U_{pos}^{att} \times s_{pos})
        9. e_{i\^parser} = v^{T} \times tanh(W_{att} \times h_{i} + U_{parser}^{att} \times s_{parser})
        10. e_{i\^dict} = v^{T} \times tanh(W_{att} \times h_{i} + U_{dict}^{att} \times s_{dict})
        11. e_{i\^kb} = v^{T} \times tanh(W_{att} \times h_{i} + U_{kb}^{att} \times s_{kb})
        12. concat all e_{ij} by 2nd dimension as e
        13. \alpha_{ij} = softmax(e_{ij})
        14. c_{i} = \alpha_{ij} \cdot s_{j}, s_{j} could be s_{word},\ s_{pos},\ s_{parser},\ s_{dict},\ s_{kb}
        15. f(h_{i}, c_{i}) = h_{i}^{T} \times c_{i}, get a scalar representing h_{i}
        16. p(i|input) = softmax(h) to get prediction result
        17. leverage cross entropy function to calculate loss
        """
        self.word = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.dictionary = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.kb_relation = tf.placeholder(shape=[cfg.batch_size, cfg.kb_relation_length], dtype=tf.int32)
        self.parser = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        self.partofspeech = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.target_id = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        # self.target_prob = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.float32)
        self.target_prob = tf.placeholder(shape=[cfg.batch_size, word_dictionary_size], dtype=tf.float32)
        self.validation_target_prob = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.float32)
        self.sess = sess

        with tf.device('/gpu:0'):
            with tf.variable_scope("psg_model"):
                self.word_embed_weight = tf.get_variable(
                    'word_emb',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.parser_embed_weight = tf.get_variable(
                    'parser_emb',
                    shape=(parser_dictionary_size, cfg.parser_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                print("self.parser_embed_weight shape is %s" % self.parser_embed_weight.get_shape())
                self.partofspeech_embed_weight = tf.get_variable(
                    'partofspeech_emb',
                    shape=(partofspeech_dictionary_size, cfg.partofspeech_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.kb_relation_embed_weight = tf.get_variable(
                    'kb_relation_emb',
                    shape=(kb_relation_dictionary_size, cfg.kb_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            # calculate initialized embedding information
            with tf.variable_scope("init_embedding"):
                word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word)
                # [cfg.batch_size, cfg.word_embedding_size]
                print('word_embed_init shape is %s' % word_embed_init.get_shape())

                # calculate parser embedding by vector addition
                parser_split_list = tf.split(self.parser, num_or_size_splits=cfg.context_window_size, axis=1)
                parser_embed_list = []
                for parser_item in parser_split_list:
                    parser_embed_list.append(tf.nn.embedding_lookup(self.parser_embed_weight, parser_item))
                parser_embed_init = tf.reduce_sum(tf.squeeze(tf.convert_to_tensor(parser_embed_list)), axis=0)
                # [cfg.batch_size, cfg.parser_embedding_size]
                print('parser_embed_init shape is %s' % parser_embed_init.get_shape())

                # calculate part-of-speech embedding
                partofspeech_embed_init = tf.nn.embedding_lookup(self.partofspeech_embed_weight, self.partofspeech)
                # [cfg.batch_size, cfg.partofspeech_embedding_size]
                print("partofspeech_embed_init shape is %s" % partofspeech_embed_init.get_shape())

                # seem dictionary state as sequential data, calculate by lstm model
                dict_desc_split_list = tf.split(self.dictionary, cfg.dict_time_step, axis=1)
                dict_desc_embed_list = []
                for dict_desc_item in dict_desc_split_list:
                    dict_desc_embed_list.append(tf.nn.embedding_lookup(self.word_embed_weight, dict_desc_item))
                dict_desc_embed_init = tf.reshape(tf.squeeze(tf.convert_to_tensor(dict_desc_embed_list)), shape=[cfg.batch_size, cfg.dict_time_step, -1])
                print("dict_desc_embed_init shape is %s" % dict_desc_embed_init.get_shape())

                cell = rnn.BasicLSTMCell(cfg.dict_lstm_hidden_size)
                init_state = cell.zero_state(cfg.batch_size, dtype=tf.float32)
                dict_desc_outputs, dict_desc_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=dict_desc_embed_init,
                                                                     initial_state=init_state, time_major=False)
                print("dict_desc_final_state.h shape is %s" % dict_desc_final_state.h.get_shape())
                print('dict_desc_outputs shape is %s, dict_desc_final_state shape[0] is %s, dict_desc_final_state shape[1] is %s'
                      % (dict_desc_outputs.get_shape(), dict_desc_final_state[0].get_shape(), dict_desc_final_state[1].get_shape()))

                # calculate kb entity embedding by vector addition
                kb_relation_split_list = tf.split(self.kb_relation, cfg.kb_relation_length, axis=1)
                kb_relation_embed_list = []
                for kb_relation_item in kb_relation_split_list:
                    kb_relation_embed_list.append(tf.nn.embedding_lookup(self.kb_relation_embed_weight, kb_relation_item))
                kb_relation_embed_init = tf.reduce_sum(tf.squeeze(tf.convert_to_tensor(kb_relation_embed_list)), axis=0)
                print("kb_relation_embed_init shape is %s" % kb_relation_embed_init.get_shape())

                word_merge_weight = tf.ones(shape=[cfg.target_lstm_hidden_size, cfg.word_embedding_size],
                                            dtype='float32')
                partofspeech_merge_weight = tf.ones(shape=[cfg.target_lstm_hidden_size, cfg.partofspeech_embedding_size],
                                                    dtype='float32')
                parser_merge_weight = tf.ones(shape=[cfg.target_lstm_hidden_size, cfg.parser_embedding_size],
                                              dtype='float32')
                dictionary_merge_weight = tf.ones(shape=[cfg.target_lstm_hidden_size, cfg.dict_lstm_hidden_size],
                                                  dtype='float32')
                kb_relation_merge_weight = tf.ones(shape=[cfg.target_lstm_hidden_size, cfg.kb_embedding_size],
                                                   dtype='float32')

            # normalize all input vectors into cfg.word_embedding_size dimension
            with tf.variable_scope("psg_attention_weight"):
                self.word_attention_weight = tf.get_variable(
                    'word_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.word_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.dictionary_attention_weight = tf.get_variable(
                    'dictionary_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.dict_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.kb_relation_attention_weight = tf.get_variable(
                    'kb_relation_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.kb_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.parser_attention_weight = tf.get_variable(
                    'parser_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.parser_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.partofspeech_attention_weight = tf.get_variable(
                    'partofspeech_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.partofspeech_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.attention_w = tf.get_variable(
                    'attention_w',
                    shape=(cfg.word_embedding_size, cfg.target_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.attention_v = tf.get_variable(
                    'attention_v',
                    shape=(1, cfg.word_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                # involve affine transformation parameters w and b to
                # align dimensions between attention result and final mapping
                self.softmax_w = tf.get_variable(
                    'softmax_w',
                    shape=(cfg.context_window_size, word_dictionary_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.softmax_b = tf.get_variable(
                    'softmax_b',
                    shape=(word_dictionary_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            with tf.variable_scope("embedding_alignment"):
                # [word_embedding_size, cfg.batch_size]
                word_attention = tf.matmul(self.word_attention_weight, tf.transpose(word_embed_init))
                print("word_attention shape is %s" % word_attention.get_shape())
                # [word_embedding_size, cfg.batch_size]
                dictionary_attention = tf.matmul(self.dictionary_attention_weight, tf.reshape(dict_desc_final_state.h,
                                                                                              shape=[cfg.dict_lstm_hidden_size, -1]))
                print("dictionary_attention shape is %s" % dictionary_attention.get_shape())
                # [word_embedding_size, cfg.batch_size]
                kb_relation_attention = tf.matmul(self.kb_relation_attention_weight, tf.reshape(kb_relation_embed_init,
                                                                                            shape=[cfg.kb_embedding_size, -1]))
                print("kb_relation_attention shape is %s" % kb_relation_attention.get_shape())
                # [word_embedding_size, cfg.batch_size]
                parser_attention = tf.matmul(self.parser_attention_weight, tf.reshape(parser_embed_init,
                                                                                            shape=[cfg.parser_embedding_size, -1]))
                print("parser_attention shape is %s" % parser_attention.get_shape())
                # [word_embedding_size, cfg.batch_size]
                partofspeech_attention = tf.matmul(self.partofspeech_attention_weight, tf.reshape(partofspeech_embed_init,
                                                                                            shape=[cfg.partofspeech_embedding_size, -1]))
                print("partofspeech_attention shape is %s" % partofspeech_attention.get_shape())

            # attention vector concatenation is safe
            '''
            # temporal solution 1
            self.nce_w = tf.get_variable(
                'nce_w',
                shape=(word_dictionary_size, cfg.word_embedding_size * 5),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            self.nce_b = tf.get_variable(
                'nce_b',
                shape=(word_dictionary_size),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            concat_tensor = tf.concat([word_attention, dictionary_attention, kb_relation_attention, parser_attention, partofspeech_attention], axis=0)
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_w, biases=self.nce_b, labels=tf.reshape(self.word, shape=[-1, 1]),
                               inputs=tf.reshape(concat_tensor, shape=[cfg.batch_size, -1]),
                               num_sampled=cfg.negative_sample_size, num_classes=word_dictionary_size))
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            self.model = tf.train.Saver()
            '''
            with tf.variable_scope("target_one_hot"):
                target_split_list = tf.split(self.target_id, cfg.context_window_size, axis=1)
                target_embed_init = []
                one_hot_list = []
                for target in target_split_list:
                    target_embed_init.append(tf.nn.embedding_lookup(self.word_embed_weight, target))
                    one_hot_tensor = tf.squeeze(tf.one_hot(target, depth=word_dictionary_size, axis=1))
                    print("one_hot_tensor shape is %s" % one_hot_tensor.get_shape())
                    one_hot_list.append(one_hot_tensor)
                # [cfg.batch_size, context_window_size, word_embedding_size]
                target_embed_init = tf.reshape(tf.squeeze(tf.convert_to_tensor(target_embed_init)), shape=[cfg.batch_size, cfg.context_window_size, -1])
                print("target_embed_init shape is %s" % target_embed_init.get_shape())
                self.one_hot_list = tf.squeeze(tf.split(tf.reshape(tf.convert_to_tensor(self.one_hot_list), shape=[cfg.batch_size, cfg.context_window_size, -1]),
                                              num_or_size_splits=cfg.batch_size))
                print("one_hot_list shape is %s" % self.one_hot_list.get_shape())

            with tf.variable_scope("target"):
                target_cell = rnn.BasicLSTMCell(cfg.target_lstm_hidden_size)
                target_init_state = target_cell.zero_state(cfg.batch_size, dtype=tf.float32)
                target_outputs, target_final_state = tf.nn.dynamic_rnn(cell=target_cell, inputs=target_embed_init,
                                                                     initial_state=target_init_state, time_major=False)
                print("target_final_state.h shape is %s" % target_final_state.h.get_shape())
                # target_outputs shape is [cfg.batch_size, cfg.context_window_size, cfg.target_lstm_hidden_size]
                print('target_outputs shape is %s, target_final_state shape[0] is %s, target_final_state shape[1] is %s'
                    % (target_outputs.get_shape(), target_final_state[0].get_shape(),
                       target_final_state[1].get_shape()))

            target_lstm_list = tf.split(target_outputs, cfg.context_window_size, axis=1)
            final_softmax_list = []
            for target_lstm in target_lstm_list:
                # target_lstm shape is [cfg.batch_size, cfg.target_lstm_hidden_size]
                target_lstm = tf.squeeze(target_lstm)
                print("target_lstm shape is %s" % target_lstm.get_shape())
                # [cfg.batch_size]
                self.e_word = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                            tf.reshape(target_lstm, shape=[cfg.target_lstm_hidden_size, -1])), word_attention))))
                print("e_word shape is %s" % self.e_word.get_shape())
                e_dictionary = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                            tf.reshape(target_lstm, shape=[cfg.target_lstm_hidden_size, -1])), dictionary_attention))))
                print("e_dictionary shape is %s" % e_dictionary.get_shape())
                e_kb_relation = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                            tf.reshape(target_final_state.h,shape=[cfg.target_lstm_hidden_size,-1])), kb_relation_attention))))
                print("e_kb_relation shape is %s" % e_kb_relation.get_shape())
                e_parser = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                            tf.reshape(target_final_state.h,shape=[cfg.target_lstm_hidden_size,-1])), parser_attention))))
                print("e_parser shape is %s" % e_parser.get_shape())
                e_partofspeech = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                            tf.reshape(target_final_state.h,shape=[cfg.target_lstm_hidden_size,-1])), partofspeech_attention))))
                print("e_partofspeech shape is %s" % e_partofspeech.get_shape())
                alpha_attention = tf.nn.softmax(tf.concat([tf.reshape(self.e_word, shape=[cfg.batch_size, -1]),
                                             tf.reshape(e_dictionary, shape=[cfg.batch_size, -1]),
                                             tf.reshape(e_kb_relation, shape=[cfg.batch_size, -1]),
                                             tf.reshape(e_parser, shape=[cfg.batch_size, -1]),
                                             tf.reshape(e_partofspeech, shape=[cfg.batch_size, -1])], axis=1), axis=1)
                # [cfg.batch_size, 5]
                print("alpha_attention shape is %s" % alpha_attention.get_shape())
                alpha_attention_tile = tf.tile(alpha_attention, [1, cfg.target_lstm_hidden_size])
                # [cfg.batch_size, 5 * cfg.target_lstm_hidden_size]
                print("alpha_attention_tile shape is %s" % alpha_attention_tile.get_shape())
                attention_concat = tf.reshape(tf.concat([tf.matmul(word_merge_weight, tf.reshape(word_embed_init, shape=[cfg.word_embedding_size, -1])),
                        tf.matmul(dictionary_merge_weight, tf.reshape(dict_desc_final_state.h, shape=[cfg.dict_lstm_hidden_size,-1])),
                        tf.matmul(kb_relation_merge_weight, tf.reshape(kb_relation_embed_init, shape=[cfg.kb_embedding_size, -1])),
                        tf.matmul(parser_merge_weight, tf.reshape(parser_embed_init, shape=[cfg.parser_embedding_size, -1])),
                        tf.matmul(partofspeech_merge_weight, tf.reshape(partofspeech_embed_init, shape=[cfg.partofspeech_embedding_size, -1]))], axis=0),
                    shape = [cfg.batch_size, -1])
                # [cfg.batch_size, 5 * cfg.target_lstm_hidden_size]
                print("attention_concat shape is %s" % attention_concat.get_shape())
                c_attention = tf.reduce_sum(tf.reshape(tf.multiply(alpha_attention_tile, attention_concat),
                            shape=[cfg.batch_size, cfg.target_lstm_hidden_size, -1]), axis=2)
                # [cfg.batch_size, cfg.target_lstm_hidden_size]
                print("c_attention shape is %s" % c_attention.get_shape())
                cur_node = tf.reduce_sum(tf.square(tf.multiply(target_lstm, c_attention)), 1)
                # [cfg.batch_size]
                print("cur_node shape is %s" % cur_node.get_shape())
                final_softmax_list.append(cur_node)
            final_softmax_list = tf.split(value=tf.transpose(tf.convert_to_tensor(final_softmax_list)), num_or_size_splits=cfg.batch_size)
            final_list = []
            for softmax_sample in final_softmax_list:
                final_list.append(tf.reshape(tf.tile(tf.squeeze(softmax_sample), multiples=[cfg.context_window_size]), shape=[cfg.context_window_size, -1]))
            final_list = tf.split(tf.reshape(tf.convert_to_tensor(final_list), shape=[cfg.batch_size, cfg.context_window_size, -1]),
                                  num_or_size_splits=cfg.batch_size)
            proj_layer_list = []
            for idx in xrange(len(final_list)):
                proj_layer_list.append(tf.matmul(tf.squeeze(final_list[idx]), self.one_hot_list[idx]))
            self.final_softmax_tensor = tf.reduce_sum(tf.convert_to_tensor(proj_layer_list), axis=1)
            print("proj_layer_list shape is %s" % self.final_softmax_tensor.get_shape())

            # NOT use softmax_w/softmax_b as additional parameters
            #final_softmax_tensor = tf.nn.bias_add(tf.matmul(tf.reshape(tf.convert_to_tensor(final_softmax_list), shape=[cfg.batch_size, -1]),
            #                                 self.softmax_w), self.softmax_b)
            #print("final_softmax_tensor shape is %s" % final_softmax_tensor.get_shape())

            final_softmax_tensor = tf.nn.softmax(self.final_softmax_tensor, axis=1)
            print("final_softmax shape is %s" % final_softmax_tensor.get_shape())
            self.cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_prob, logits=final_softmax_tensor))
            print("cross_entropy_loss shape is %s" % self.cross_entropy_loss.get_shape())
            self.opt = tf.train.AdamOptimizer().minimize(self.cross_entropy_loss)
            self.model = tf.train.Saver()

            #final_softmax_tensor = tf.nn.bias_add(tf.matmul(tf.reshape(concat_tensor, shape=[cfg.batch_size, -1]),
            #                                                tf.reshape(self.nce_w, shape=[-1, word_dictionary_size])), self.nce_b)
            # [cfg.batch_size, word_dictionary_size]
            #print("final_softmax_tensor shape is %s" % final_softmax_tensor.get_shape())

            # get prediction result from softmax according by target_id information
            final_softmax_tensor_list = tf.split(final_softmax_tensor, num_or_size_splits=cfg.batch_size)
            target_id_list = tf.split(self.target_id, num_or_size_splits=cfg.batch_size)
            comparison_list = []
            for idx, final_softmax_element in enumerate(final_softmax_tensor_list):
                comparison_list.append(tf.squeeze(tf.nn.embedding_lookup(
                    tf.reshape(final_softmax_element, shape=[word_dictionary_size, -1]), target_id_list[idx])))
            print("comparison_list shape is %s" % tf.convert_to_tensor(comparison_list).get_shape())
            comparison = tf.equal(tf.argmax(tf.convert_to_tensor(comparison_list), axis=1), tf.argmax(self.validation_target_prob, axis=1))
            print("comparison shape is %s" % comparison.get_shape())
            self.accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))
            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word, dictionary, kb_relation, parser, partofspeech, target_id, target_prob):
        return self.sess.run([self.opt, self.cross_entropy_loss, self.final_softmax_tensor], feed_dict={
            self.word: word,
            self.dictionary: dictionary,
            self.kb_relation: kb_relation,
            self.parser: parser,
            self.partofspeech: partofspeech,
            self.target_id: target_id,
            self.target_prob: target_prob})

    def validate(self, word, dictionary, kb_relation, parser, partofspeech, target_id, validation_target_prob):
        return self.sess.run(self.accuracy, feed_dict={
            self.word: word,
            self.dictionary: dictionary,
            self.kb_relation: kb_relation,
            self.parser: parser,
            self.partofspeech: partofspeech,
            self.target_id: target_id,
            self.validation_target_prob: validation_target_prob
        })

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

    def get_word_emb(self):
        return self.sess.run(self.word_embed_weight, feed_dict={})

if __name__ == '__main__':
    if len(sys.argv) < 14:
        print("probabilistic_skip_gram <target> <word_dict> <context> <part-of-speech> <part-of-speech_dict> <parser> "
              "<parser_dict> <dictionary desc> <kb entity> <kb_entity_dict> <word count dict> <word coocur dict> <word emb output>")
        sys.exit()
    [word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size,
     kb_relation_dictionary_size] = load_sample(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                                                sys.argv[6], sys.argv[7], sys.argv[8],
                                                sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])
    print("word_dictionary_size is %d, partofspeech_dictionary_size is %d, parser_dictionary_size is %d,"
          "kb_relation_dictionary_size is %d" % (
          word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size, kb_relation_dictionary_size))

    total_sample_size = target.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_size_fake = int(total_batch_size * 1)

    print('total_batch_size is %d, train_set_size is %d' %
          (total_batch_size, train_set_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PSGModelObj = PSGModel(sess)
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_test_summary_writer_path, sess.graph)

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print(k)

        trainable = False
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(total_batch_size):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                context_prob_info = context_prob[i*cfg.batch_size:(i+1)*cfg.batch_size]
                context_prob_1st_dim = context_prob_info.shape[0]
                context_prob_2nd_dim = context_prob_info.shape[1]
                # context_prob_tmp's shape is [cfg.batch_size, word_dictionary_size]
                context_prob_tmp = np.zeros(shape=[context_prob_1st_dim, word_dictionary_size])
                for idx, val in enumerate(target[i*cfg.batch_size:(i+1)*cfg.batch_size]):
                    context_prob_tmp[idx:val.astype(np.int)] = 1
                _, iter_loss, word_embed_weight = PSGModelObj.train(target[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 dict_desc[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 kb_entity[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 parser[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 pos[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 context[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 context_prob_tmp)
                final_score_list = []
                for j in np.nonzero(word_embed_weight[0]):
                    final_score_list.append(word_embed_weight[0][j])
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, np.sum(loss_sum) / cfg.batch_size / total_batch_size))
            train_loss = PSGModelObj.get_loss_summary(np.sum(loss_sum) / cfg.batch_size / total_batch_size)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for j in range(total_batch_size):
                '''
                context_prob_info = context_prob[i * cfg.batch_size:(i + 1) * cfg.batch_size]
                context_prob_1st_dim = context_prob_info.shape[0]
                context_prob_2nd_dim = context_prob_info.shape[1]
                context_prob_tmp = np.zeros(shape=[context_prob_1st_dim, word_dictionary_size])
                for idx, val in enumerate(target[i*cfg.batch_size:(i+1)*cfg.batch_size]):
                    context_prob_tmp[idx:val.astype(np.int)] = 1
                '''
                iter_accuracy = PSGModelObj.validate(target[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     dict_desc[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     kb_entity[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     parser[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     pos[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     context[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     context_prob[i * cfg.batch_size:(i + 1) * cfg.batch_size])
                accuracy += iter_accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy / total_batch_size / cfg.batch_size))
            test_accuracy = PSGModelObj.get_accuracy_summary(accuracy / total_batch_size / cfg.batch_size)
            test_writer.add_summary(test_accuracy, epoch_index + 1)

        embed_weight = PSGModelObj.get_word_emb()
        output_embed_file = open(sys.argv[1], 'w')
        for embed_item in embed_weight:
            embed_list = list(embed_item)
            embed_list = [str(item) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()

        sess.close()