# -*- coding: utf-8 -*-
# @Time        : 2018/11/12 16:43
# @Author      : panxiaotong
# @Description : MIPSG-II, just use target's last status calculated by lstm during attention alignment

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tensorflow.contrib import rnn
import tensorflow as tf
from config.config import cfg
import numpy as np

word_dictionary_size = 0
parser_dictionary_size = 0
partofspeech_dictionary_size = 0
kb_relation_dictionary_size = 0

def load_sample(word_dict_file, pos_dict_file, parser_dict_file, kb_entity_dict_file, word_count_file, word_coocur_file):
    word_dictionary_size = len(open(word_dict_file).readlines()) + 1
    partofspeech_dictionary_size = len(open(pos_dict_file).readlines()) + 1
    parser_dictionary_size = len(open(parser_dict_file).readlines()) + 1
    kb_relation_dictionary_size = len(open(kb_entity_dict_file).readlines()) + 1
    word_count_dict = {}
    word_coocur_dict = {}
    with open(word_count_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_count_dict[int(elements[0])] = int(elements[1])
        f.close()
    with open(word_coocur_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word_coocur_dict[elements[0]] = int(elements[1])
        f.close()

    return [word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size, kb_relation_dictionary_size,
            word_count_dict, word_coocur_dict]

class MIPSG2():
    def __init__(self, sess):
        """
        :param sess: session passed by main entry function
        :algorithm
        1. define W_{word_id},\ W_{pos},\ W_{parser},\ W_{kb} embedding learnable variables
        2. aggregate all parser information of one word by vector addition, h_{parser}
        3. calculate dictionary information by lstm, leverage last state as dictionary state, h_{dict}
        4. aggretate all knowledge-base entities of one word by vector addition, h_{kb}
        5. define 4 U variables in attention, U_{word_id}^{att},\ U_{pos}^{att},\ U_{parser}^{att},\ U_{dict}^{att},\ U_{kb}^{att}
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
        self.dictionary = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.kb_relation = tf.placeholder(shape=[cfg.batch_size, cfg.kb_relation_length], dtype=tf.int32)
        self.parser = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        self.partofspeech = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.target_id = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        self.target_prob = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.float32)
        self.sampled_candidates = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=tf.int32)
        self.sampled_expected_count = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=tf.float32)
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

            # get initialized embedding information
            with tf.variable_scope("init_embedding"):
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

                cell = rnn.BasicLSTMCell(cfg.dict_lstm_hidden_size, name="dict_rnn")
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

            '''
            c_attention = tf.concat([parser_embed_init,
                                     partofspeech_embed_init,
                                     dict_desc_final_state.h,
                                     kb_relation_embed_init], axis=1)
            print("c_attention shape is %s" % c_attention.get_shape())
            self.proj_w = tf.get_variable(
                'proj_w',
                shape=(word_dictionary_size,
                       cfg.parser_embedding_size + cfg.partofspeech_embedding_size +
                       cfg.dict_lstm_hidden_size + cfg.kb_embedding_size),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            print("proj_w shape is %s" % self.proj_w.get_shape())
            self.proj_b = tf.get_variable(
                'proj_b',
                shape=(word_dictionary_size),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            print("proj_b shape is %s" % self.proj_b.get_shape())
            self.cross_entropy_loss = tf.nn.nce_loss(weights=self.proj_w,
                                                     biases=self.proj_b,
                                                     labels=self.target_id,
                                                     inputs=c_attention,
                                                     num_sampled=cfg.negative_sample_size,
                                                     num_classes=word_dictionary_size,
                                                     num_true=cfg.context_window_size,
                                                     sampled_values=[tf.reshape(self.sampled_candidates, shape=[-1]),
                                                                     self.target_prob,
                                                                     tf.reshape(self.sampled_expected_count, shape=[-1])])

            '''
            # normalize all input vectors into cfg.attention_size dimension
            with tf.variable_scope("psg_attention_weight"):
                self.dictionary_attention_weight = tf.get_variable(
                    'dictionary_attention_weight',
                    shape=(cfg.attention_size, cfg.dict_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.kb_relation_attention_weight = tf.get_variable(
                    'kb_relation_attention_weight',
                    shape=(cfg.attention_size, cfg.kb_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.parser_attention_weight = tf.get_variable(
                    'parser_attention_weight',
                    shape=(cfg.attention_size, cfg.parser_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.partofspeech_attention_weight = tf.get_variable(
                    'partofspeech_attention_weight',
                    shape=(cfg.attention_size, cfg.partofspeech_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.attention_w = tf.get_variable(
                    'attention_w',
                    shape=(cfg.attention_size, cfg.target_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.attention_v = tf.get_variable(
                    'attention_v',
                    shape=(1, cfg.attention_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            with tf.variable_scope("embedding_alignment"):
                dictionary_attention = tf.matmul(self.dictionary_attention_weight, tf.reshape(dict_desc_final_state.h,
                                                                                              shape=[cfg.dict_lstm_hidden_size, -1]))
                # [attention_size, cfg.batch_size]
                print("dictionary_attention shape is %s" % dictionary_attention.get_shape())
                kb_relation_attention = tf.matmul(self.kb_relation_attention_weight, tf.reshape(kb_relation_embed_init,
                                                                                            shape=[cfg.kb_embedding_size, -1]))
                # [attention_size, cfg.batch_size]
                print("kb_relation_attention shape is %s" % kb_relation_attention.get_shape())
                parser_attention = tf.matmul(self.parser_attention_weight, tf.reshape(parser_embed_init,
                                                                                            shape=[cfg.parser_embedding_size, -1]))
                # [attention_size, cfg.batch_size]
                print("parser_attention shape is %s" % parser_attention.get_shape())
                partofspeech_attention = tf.matmul(self.partofspeech_attention_weight, tf.reshape(partofspeech_embed_init,
                                                                                            shape=[cfg.partofspeech_embedding_size, -1]))
                # [attention_size, cfg.batch_size]
                print("partofspeech_attention shape is %s" % partofspeech_attention.get_shape())
            # expand self.target_id into one hot space
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
                one_hot_list = tf.squeeze(tf.split(tf.reshape(tf.convert_to_tensor(one_hot_list), shape=[cfg.batch_size, cfg.context_window_size, -1]),
                                              num_or_size_splits=cfg.batch_size))
                # [cfg.batch_size, cfg.context_window_size, word_dictionary_size]
                print("one_hot_list shape is %s" % one_hot_list.get_shape())

            # target lstm calculation
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

            e_dictionary = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                    tf.reshape(target_final_state.h, shape=[cfg.target_lstm_hidden_size, -1])), dictionary_attention))))
            print("e_dictionary shape is %s" % e_dictionary.get_shape())
            e_kb_relation = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                    tf.reshape(target_final_state.h, shape=[cfg.target_lstm_hidden_size,-1])), kb_relation_attention))))
            print("e_kb_relation shape is %s" % e_kb_relation.get_shape())
            e_parser = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                    tf.reshape(target_final_state.h,shape=[cfg.target_lstm_hidden_size,-1])), parser_attention))))
            print("e_parser shape is %s" % e_parser.get_shape())
            e_partofspeech = tf.squeeze(tf.matmul(self.attention_v, tf.nn.tanh(tf.add(tf.matmul(self.attention_w,
                    tf.reshape(target_final_state.h,shape=[cfg.target_lstm_hidden_size,-1])), partofspeech_attention))))
            print("e_partofspeech shape is %s" % e_partofspeech.get_shape())
            alpha_attention = tf.nn.softmax(tf.concat([tf.reshape(e_dictionary, shape=[cfg.batch_size, -1]),
                                                       tf.reshape(e_kb_relation, shape=[cfg.batch_size, -1]),
                                                       tf.reshape(e_parser, shape=[cfg.batch_size, -1]),
                                                       tf.reshape(e_partofspeech, shape=[cfg.batch_size, -1])], axis=1),
                                            axis=1)
            alpha_attention_tile = tf.tile(alpha_attention, [1, cfg.attention_size])
            # [cfg.batch_size, 4 * cfg.attention_size]
            print("alpha_attention_tile shape is %s" % alpha_attention_tile.get_shape())
            # [cfg.batch_size, 4]
            print("alpha_attention shape is %s" % alpha_attention.get_shape())
            c_attention = tf.reduce_sum(tf.reshape(tf.multiply(tf.concat([tf.reshape(dictionary_attention, shape=[cfg.batch_size, -1]),
                                     tf.reshape(kb_relation_attention, shape=[cfg.batch_size, -1]),
                                     tf.reshape(parser_attention, shape=[cfg.batch_size, -1]),
                                     tf.reshape(partofspeech_attention, shape=[cfg.batch_size, -1])], axis=1),
                                      alpha_attention_tile), shape=[cfg.batch_size, 4, -1]), axis=1)
            # [cfg.batch_size, cfg.attention_size]
            print("c_attention shape is %s" % c_attention.get_shape())
            # involve affine transformation parameters w and b to
            # align dimensions between attention result and final mapping
            self.proj_w = tf.get_variable(
                'proj_w',
                shape=(word_dictionary_size, cfg.attention_size),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            self.proj_b = tf.get_variable(
                'proj_b',
                shape=(word_dictionary_size),
                initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                dtype='float32'
            )
            self.cross_entropy_loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.proj_w,
                                                     biases=self.proj_b,
                                                     labels=self.target_id,
                                                     inputs=c_attention,
                                                     num_sampled=cfg.negative_sample_size,
                                                     num_classes=word_dictionary_size,
                                                     num_true=cfg.context_window_size,
                                                     sampled_values=[tf.reshape(self.sampled_candidates, shape=[-1]),
                                                                     self.target_prob,
                                                                     tf.reshape(self.sampled_expected_count,
                                                                                shape=[-1])]))
            self.opt = tf.train.AdamOptimizer().minimize(self.cross_entropy_loss)
            self.model = tf.train.Saver()

            softmax_layer = tf.nn.softmax(tf.nn.bias_add(
                tf.matmul(c_attention, tf.reshape(self.proj_w, shape=[cfg.attention_size, -1])), self.proj_b), axis=1)
            # [cfg.batch_size, word_dictionary_size]
            print("softmax_layer shape is %s" % softmax_layer.get_shape())
            # get prediction result from softmax according by target_id information
            final_softmax_tensor_list = tf.split(softmax_layer, num_or_size_splits=cfg.batch_size)
            sampled_candidates_list = tf.split(self.sampled_candidates, num_or_size_splits=cfg.batch_size)
            target_id_list = tf.split(self.target_id, num_or_size_splits=cfg.batch_size)
            comparison_list = []
            for idx, final_softmax_element in enumerate(final_softmax_tensor_list):
                comparison_list.append(tf.concat(
                    [tf.squeeze(tf.nn.embedding_lookup(tf.reshape(final_softmax_element, shape=[word_dictionary_size, -1]), target_id_list[idx])),
                    tf.squeeze(tf.nn.embedding_lookup(tf.reshape(final_softmax_element, shape=[word_dictionary_size, -1]), sampled_candidates_list[idx]))], axis=0))
            # [cfg.batch_size, cfg.context_window_size + cfg.negative_sample_size]
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

    def train(self, dictionary, kb_relation, parser, partofspeech, target_id, target_prob, sampled_candidates, sampled_expected_count):
        return self.sess.run([self.opt, self.cross_entropy_loss], feed_dict={
            self.dictionary: dictionary,
            self.kb_relation: kb_relation,
            self.parser: parser,
            self.partofspeech: partofspeech,
            self.target_id: target_id,
            self.target_prob: target_prob,
            self.sampled_candidates: sampled_candidates,
            self.sampled_expected_count: sampled_expected_count
        })

    def validate(self, dictionary, kb_relation, parser, partofspeech, target_id, validation_target_prob, sampled_candidates):
        return self.sess.run(self.accuracy, feed_dict={
            self.dictionary: dictionary,
            self.kb_relation: kb_relation,
            self.parser: parser,
            self.partofspeech: partofspeech,
            self.target_id: target_id,
            self.validation_target_prob: validation_target_prob,
            self.sampled_candidates: sampled_candidates
        })

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

    def get_word_emb(self):
        return self.sess.run(self.word_embed_weight, feed_dict={})

    def get_parser_emb(self):
        return self.sess.run(self.parser_embed_weight, feed_dict={})

    def get_partofspeech_emb(self):
        return self.sess.run(self.partofspeech_embed_weight, feed_dict={})

    def get_kb_emb(self):
        return self.sess.run(self.kb_relation_embed_weight, feed_dict={})

if __name__ == '__main__':
    if len(sys.argv) < 18:
        print("MIPSG2 <target> <word_dict> <context> <part-of-speech> <part-of-speech_dict> <parser> "
              "<parser_dict> <dictionary desc> <kb entity> <kb_entity_dict> <word count dict> <word coocur dict> <word emb output>"
              "<negative sample file> <parser emb output> <partofspeech emb output> <kb emb output>")
        sys.exit()
    [word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size,
     kb_relation_dictionary_size, word_count_dict, word_coocur_dict] = load_sample(
        sys.argv[2], sys.argv[5], sys.argv[7], sys.argv[10], sys.argv[11], sys.argv[12])
    print("word_dictionary_size is %d, partofspeech_dictionary_size is %d, parser_dictionary_size is %d,"
          "kb_relation_dictionary_size is %d" % (
          word_dictionary_size, partofspeech_dictionary_size, parser_dictionary_size, kb_relation_dictionary_size))

    total_sample_size = len(open(sys.argv[1]).readlines())
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)

    print('total_batch_size is %d, train_set_size is %d' %
          (total_batch_size, train_set_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PSGModelObj = MIPSG2(sess)
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_test_summary_writer_path, sess.graph)

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print(k)
        trainable = False

        prev_avg_accu = 0.0
        cur_avg_accu = 0.0
        prev_loss = 0.0
        cur_loss = 0.0
        max_accu = 0.0
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            target_file = open(sys.argv[1], 'r')
            context_file = open(sys.argv[3], 'r')
            pos_file = open(sys.argv[4], 'r')
            parser_file = open(sys.argv[6], 'r')
            dict_desc_file = open(sys.argv[8], 'r')
            kb_entity_file = open(sys.argv[9], 'r')
            neg_file = open(sys.argv[13], 'r')
            for iter in range(train_set_size):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True

                # prepare train data
                counter = 0
                target = []
                for line in target_file:
                    target.append(line.strip('\r\n'))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                target = np.asarray(target, dtype=np.int32)

                counter = 0
                dict_desc = []
                for line in dict_desc_file:
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
                        for j in range(remain_len):
                            dict_desc_list.append(0)
                    dict_desc.append(dict_desc_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                dict_desc = np.asarray(dict_desc, dtype=np.int32)

                counter = 0
                kb_entity = []
                for line in kb_entity_file:
                    kb_entity_list = line.strip('\r\n').split(',')
                    if len(kb_entity_list) > cfg.kb_relation_length:
                        kb_entity_list = kb_entity_list[0:cfg.kb_relation_length]
                    else:
                        supplement_kb_size = cfg.kb_relation_length - len(kb_entity_list)
                        for j in range(supplement_kb_size):
                            kb_entity_list.append("0")
                    kb_entity.append(kb_entity_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                kb_entity = np.asarray(kb_entity, dtype=np.int32)

                counter = 0
                parser = []
                for line in parser_file:
                    parser_list = line.strip('\r\n').replace(' ', '').split(',')
                    if len(parser_list) < cfg.context_window_size:
                        supplement_parser_size = cfg.context_window_size - len(parser_list)
                        for j in range(supplement_parser_size):
                            parser_list.append(0)
                    parser.append(parser_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                parser = np.asarray(parser, dtype=np.int32)

                counter = 0
                pos = []
                for line in pos_file:
                    pos.append(line.strip('\r\n'))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                pos = np.asarray(pos, dtype=np.int32)

                counter = 0
                context = []
                context_prob = []
                for idx, line in enumerate(context_file):
                    context_word_ids = line.strip('\r\n').split(',')
                    context.append(context_word_ids)
                    sub_context_prob = []
                    # word id not in dictionary
                    if target[idx] not in word_count_dict:
                        for j in range(len(context_word_ids)):
                            sub_context_prob.append(1.0 / float(cfg.context_window_size))
                    else:
                        accumulate_count = 0
                        for co_word in context_word_ids:
                            co_name1 = str(target[idx]) + cfg.coocur_separator + co_word
                            co_name2 = co_word + cfg.coocur_separator + str(target[idx])
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
                            for j in range(len(context_word_ids)):
                                sub_context_prob.append(1.0 / float(cfg.context_window_size))
                        else:
                            sub_context_prob = [float(item) / float(accumulate_count) for item in sub_context_prob]
                    context_prob.append(np.power(sub_context_prob, cfg.normalize_value))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                context = np.asarray(context, dtype=np.int32)
                context_prob = np.asarray(context_prob, dtype=np.float32)

                counter = 0
                sampled_candidates = np.zeros(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=np.int32)
                sampled_expected_count = np.zeros(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=np.float32)
                for idx, line in enumerate(neg_file):
                    elements = [int(item) for item in line.strip('\r\n').split(',')]
                    sampled_candidates[idx] = np.asarray(elements, dtype=np.int32)
                    sampled_expected_count[idx] = np.full(shape=[cfg.negative_sample_size],
                                                          fill_value=(1.0 - np.sum(
                                                              context_prob[idx])) / cfg.negative_sample_size,
                                                          dtype=np.float32)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break

                _, iter_loss = PSGModelObj.train(
                                                 dict_desc,
                                                 kb_entity,
                                                 parser,
                                                 pos,
                                                 context,
                                                 context_prob,
                                                 sampled_candidates,
                                                 sampled_expected_count)
                loss_sum += iter_loss
            loss_sum /= train_set_size
            print("epoch_index %d, loss is %f" % (epoch_index, loss_sum))
            train_loss = PSGModelObj.get_loss_summary(loss_sum)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for iter in range(total_batch_size - train_set_size):
                # prepare train data
                counter = 0
                target = []
                for line in target_file:
                    target.append(line.strip('\r\n'))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                target = np.asarray(target, dtype=np.int32)

                counter = 0
                dict_desc = []
                for line in dict_desc_file:
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
                        for j in range(remain_len):
                            dict_desc_list.append(0)
                    dict_desc.append(dict_desc_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                dict_desc = np.asarray(dict_desc, dtype=np.int32)

                counter = 0
                kb_entity = []
                for line in kb_entity_file:
                    kb_entity_list = line.strip('\r\n').split(',')
                    if len(kb_entity_list) > cfg.kb_relation_length:
                        kb_entity_list = kb_entity_list[0:cfg.kb_relation_length]
                    else:
                        supplement_kb_size = cfg.kb_relation_length - len(kb_entity_list)
                        for j in range(supplement_kb_size):
                            kb_entity_list.append("0")
                    kb_entity.append(kb_entity_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                kb_entity = np.asarray(kb_entity, dtype=np.int32)

                counter = 0
                parser = []
                for line in parser_file:
                    parser_list = line.strip('\r\n').replace(' ', '').split(',')
                    if len(parser_list) < cfg.context_window_size:
                        supplement_parser_size = cfg.context_window_size - len(parser_list)
                        for j in range(supplement_parser_size):
                            parser_list.append(0)
                    parser.append(parser_list)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                parser = np.asarray(parser, dtype=np.int32)

                counter = 0
                pos = []
                for line in pos_file:
                    pos.append(line.strip('\r\n'))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                pos = np.asarray(pos, dtype=np.int32)

                counter = 0
                context = []
                context_prob = []
                for idx, line in context_file:
                    context_word_ids = line.strip('\r\n').split(',')
                    context.append(context_word_ids)
                    sub_context_prob = []
                    # word id not in dictionary
                    if target[idx] not in word_count_dict:
                        for i in range(len(context_word_ids)):
                            sub_context_prob.append(1.0 / float(cfg.context_window_size))
                    else:
                        accumulate_count = 0
                        for co_word in context_word_ids:
                            co_name1 = str(target[idx]) + cfg.coocur_separator + co_word
                            co_name2 = co_word + cfg.coocur_separator + str(target[idx])
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
                    context_prob.append(np.power(sub_context_prob, cfg.normalize_value))
                    counter += 1
                    if counter >= cfg.batch_size:
                        break
                context = np.asarray(context, dtype=np.int32)
                context_prob = np.asarray(context_prob, dtype=np.float32)

                counter = 0
                sampled_candidates = np.zeros(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=np.int32)
                for idx, line in neg_file:
                    elements = [int(item) for item in line.strip('\r\n').split(',')]
                    sampled_candidates[idx] = np.asarray(elements, dtype=np.int32)
                    counter += 1
                    if counter >= cfg.batch_size:
                        break

                iter_accuracy = PSGModelObj.validate(
                                                     dict_desc,
                                                     kb_entity,
                                                     parser,
                                                     pos,
                                                     context,
                                                     context_prob,
                                                     sampled_candidates)
                accuracy += iter_accuracy
            accuracy /= (total_batch_size - train_set_size)
            if max_accu < accuracy:
                max_accu = accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy))
            if epoch_index < cfg.early_stop_iter:
                prev_avg_accu += accuracy
                prev_loss += loss_sum
            elif epoch_index % cfg.early_stop_iter == 0 and epoch_index / cfg.early_stop_iter > 1:
                if cur_avg_accu <= prev_avg_accu and prev_loss <= cur_loss:
                    print("training converge in epoch %d: prev_accu %f, cur_accu %f, prev_loss %f, cur_loss %f" %
                          (epoch_index, prev_avg_accu, cur_avg_accu, prev_loss, cur_loss))
                    break
                else:
                    prev_avg_accu = cur_avg_accu
                    cur_avg_accu = accuracy
                    prev_loss = cur_loss
                    cur_loss = loss_sum
            else:
                cur_avg_accu += accuracy
                cur_loss += loss_sum
            test_accuracy = PSGModelObj.get_accuracy_summary(accuracy)
            test_writer.add_summary(test_accuracy, epoch_index + 1)

            target_file.close()
            context_file.close()
            pos_file.close()
            parser_file.close()
            dict_desc_file.close()
            kb_entity_file.close()
            neg_file.close()
        print('max_accu is %f' % max_accu)

        word_embed_weight = PSGModelObj.get_word_emb()
        output_embed_file = open(sys.argv[14], 'w')
        for embed_item in word_embed_weight:
            embed_list = list(embed_item)
            embed_list = [str(item) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()

        parser_embed_weight = PSGModelObj.get_parser_emb()
        output_embed_file = open(sys.argv[15], 'w')
        for embed_item in parser_embed_weight:
            embed_list = list(embed_item)
            embed_list = [(str(item)) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()

        partofspeech_embed_weight = PSGModelObj.get_partofspeech_emb()
        output_embed_file = open(sys.argv[16], 'w')
        for embed_item in partofspeech_embed_weight:
            embed_list = list(embed_item)
            embed_list = [(str(item)) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()

        kb_embed_weight = PSGModelObj.get_kb_emb()
        output_embed_file = open(sys.argv[17], 'w')
        for embed_item in kb_embed_weight:
            embed_list = list(embed_item)
            embed_list = [(str(item)) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()

        sess.close()