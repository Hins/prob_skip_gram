# -*- coding: utf-8 -*-
# @Time        : 2018/9/4 17:26
# @Author      : panxiaotong
# @Description : upgrade PGM with merging weighted linguistic embedding by CNN

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
pos = []
parser = []
dict_desc = []
kb_entity = []
def load_sample(target_file, context_file, pos_file, parser_file, dict_desc_file, kb_entity_file):
    global target, context, pos, parser, dict_desc, kb_entity
    with open(target_file, 'r') as f:
        for line in f:
            target.appen(line.replace('\r\n'))
        f.close()
    with open(context_file, 'r') as f:
        for line in f:
            context.append(line.replace('\r\n').split(','))
        f.close()
    with open(pos_file, 'r') as f:
        for line in f:
            pos.append(line.replace('\r\n'))
        f.close()
    with open(parser_file, 'r') as f:
        for line in f:
            parser.append(line.replace('\r\n').split(','))
        f.close()
    with open(dict_desc_file, 'r') as f:
        for line in f:
            elements = line.replace('\r\n').split(',')
            desc_len = len(elements)
            dict_desc_list = []
            if desc_len >= cfg.dict_time_step:
                for index, item in enumerate(elements):
                    if index >= cfg.dict_time_step:
                        break
                    dict_desc_list.append(item)
            else:
                dict_desc_list = elements.copy()
                remain_len = cfg.dict_time_step - desc_len
                for i in range(remain_len):
                    dict_desc_list.append(0)
            dict_desc.append(dict_desc_list)
        f.close()
    with open(kb_entity_file, 'r') as f:
        for line in f:
            kb_entity.append(line.replace('\r\n'))
        f.close()

    target = np.asarray(target)
    context = np.asarray(context)
    pos = np.asarray(pos)
    parser = np.asarray(parser)
    dict_desc = np.asarray(dict_desc)
    kb_entity = np.asarray(kb_entity)

class PSGModel():
    def __init__(self, sess):
        self.word = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.dictionary = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.kb_relation = tf.placeholder(shape=[cfg.batch_size, cfg.kb_relation_length], dtype=tf.int32)
        self.parser = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        self.partofspeech = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.target_id = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.int32)
        self.target_prob = tf.placeholder(shape=[cfg.batch_size, cfg.context_window_size], dtype=tf.float32)
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


            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word)
            # [cfg.batch_size, cfg.word_embedding_size]
            print('word_embed_init shape is %s' % word_embed_init.get_shape())

            parser_split_list = tf.split(self.parser, num_or_size_splits=cfg.context_window_size, axis=1)
            parser_embed_list = []
            for parser_item in parser_split_list:
                parser_embed_list.append(tf.nn.embedding_lookup(self.parser_embed_weight, parser_item))
            parser_embed_init = tf.reduce_sum(tf.squeeze(tf.convert_to_tensor(parser_embed_list)), axis=0)
            # [cfg.batch_size, cfg.parser_embedding_size]
            print('parser_embed_init shape is %s' % parser_embed_init.get_shape())

            partofspeech_embed_init = tf.nn.embedding_lookup(self.partofspeech_embed_weight, self.partofspeech)
            # [cfg.batch_size, cfg.partofspeech_embedding_size]
            print("partofspeech_embed_init shape is %s" % partofspeech_embed_init.get_shape())

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

            kb_relation_split_list = tf.split(self.kb_relation, cfg.kb_relation_length, axis=1)
            kb_relation_embed_list = []
            for kb_relation_item in kb_relation_split_list:
                kb_relation_embed_list.append(tf.nn.embedding_lookup(self.kb_relation_embed_weight, kb_relation_item))
            kb_relation_embed_init = tf.reduce_sum(tf.squeeze(tf.convert_to_tensor(kb_relation_embed_list)), axis=0)
            print("kb_relation_embed_init shape is %s" % kb_relation_embed_init.get_shape())

            with tf.variable_scope("psg_attention_weight"):
                self.word_attention_weight = tf.get_variable(
                    'word_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.word_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.word_merge_weight = tf.get_variable(
                    'word_merge_weight',
                    shape=(cfg.target_lstm_hidden_size, cfg.word_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.dictionary_attention_weight = tf.get_variable(
                    'dictionary_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.dict_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.dictionary_merge_weight = tf.get_variable(
                    'dictionary_merge_weight',
                    shape=(cfg.target_lstm_hidden_size, cfg.dict_lstm_hidden_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.kb_relation_attention_weight = tf.get_variable(
                    'kb_relation_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.kb_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.kb_relation_merge_weight = tf.get_variable(
                    'kb_relation_merge_weight',
                    shape=(cfg.target_lstm_hidden_size, cfg.kb_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.parser_attention_weight = tf.get_variable(
                    'parser_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.parser_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.parser_merge_weight = tf.get_variable(
                    'parser_merge_weight',
                    shape=(cfg.target_lstm_hidden_size, cfg.parser_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.partofspeech_attention_weight = tf.get_variable(
                    'partofspeech_attention_weight',
                    shape=(cfg.word_embedding_size, cfg.partofspeech_embedding_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.partofspeech_merge_weight = tf.get_variable(
                    'partofspeech_merge_weight',
                    shape=(cfg.target_lstm_hidden_size, cfg.partofspeech_embedding_size),
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

            word_attention = tf.matmul(self.word_attention_weight, word_embed_init)
            print("word_attention shape is %s" % word_attention.get_shape())
            dictionary_attention = tf.matmul(self.dictionary_attention_weight, tf.reshape(dict_desc_final_state.h,
                                                                                          shape=[cfg.dict_lstm_hidden_size, -1]))
            print("dictionary_attention shape is %s" % dictionary_attention.get_shape())
            kb_relation_attention = tf.matmul(self.kb_relation_attention_weight, tf.reshape(kb_relation_embed_init,
                                                                                        shape=[cfg.kb_embedding_size, -1]))
            print("kb_relation_attention shape is %s" % kb_relation_attention.get_shape())
            parser_attention = tf.matmul(self.parser_attention_weight, tf.reshape(parser_embed_init,
                                                                                        shape=[cfg.parser_embedding_size, -1]))
            print("parser_attention shape is %s" % parser_attention.get_shape())
            partofspeech_attention = tf.matmul(self.partofspeech_attention_weight, tf.reshape(partofspeech_embed_init,
                                                                                        shape=[cfg.partofspeech_embedding_size, -1]))
            print("partofspeech_attention shape is %s" % partofspeech_attention.get_shape())

            target_split_list = tf.split(self.target_id, cfg.context_window_size, axis=1)
            target_embed_init = []
            for target in target_split_list:
                target_embed_init.append(tf.nn.embedding_lookup(self.word_embed_weight, target))
            target_embed_init = tf.reshape(tf.squeeze(tf.convert_to_tensor(target_embed_init)), shape=[cfg.batch_size, cfg.context_window_size, -1])
            print("target_embed_init shape is %s" % target_embed_init.get_shape())

            with tf.variable_scope("target"):
                target_cell = rnn.BasicLSTMCell(cfg.target_lstm_hidden_size)
                target_init_state = target_cell.zero_state(cfg.batch_size, dtype=tf.float32)
                target_outputs, target_final_state = tf.nn.dynamic_rnn(cell=target_cell, inputs=target_embed_init,
                                                                     initial_state=target_init_state, time_major=False)
                print("target_final_state.h shape is %s" % target_final_state.h.get_shape())
                print('target_outputs shape is %s, target_final_state shape[0] is %s, target_final_state shape[1] is %s'
                    % (target_outputs.get_shape(), target_final_state[0].get_shape(),
                       target_final_state[1].get_shape()))

            target_lstm_list = tf.split(target_outputs, cfg.context_window_size, axis=1)
            final_softmax_list = []
            for target_lstm in target_lstm_list:
                target_lstm = tf.squeeze(target_lstm)
                print("target_lstm shape is %s" % target_lstm.get_shape())
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
                print("alpha_attention shape is %s" % alpha_attention.get_shape())
                alpha_attention_tile = tf.tile(alpha_attention, [1, cfg.target_lstm_hidden_size])
                print("alpha_attention_tile shape is %s" % alpha_attention_tile.get_shape())
                attention_concat = tf.reshape(tf.concat([tf.matmul(self.word_merge_weight, tf.reshape(word_embed_init, shape=[cfg.word_embedding_size, -1])),
                        tf.matmul(self.dictionary_merge_weight, tf.reshape(dict_desc_final_state.h, shape=[cfg.dict_lstm_hidden_size,-1])),
                        tf.matmul(self.kb_relation_merge_weight, tf.reshape(kb_relation_embed_init, shape=[cfg.kb_embedding_size, -1])),
                        tf.matmul(self.parser_merge_weight, tf.reshape(parser_embed_init, shape=[cfg.parser_embedding_size, -1])),
                        tf.matmul(self.partofspeech_merge_weight, tf.reshape(partofspeech_embed_init, shape=[cfg.partofspeech_embedding_size, -1]))], axis=0),
                    shape = [cfg.batch_size, -1])
                print("attention_concat shape is %s" % attention_concat.get_shape())
                c_attention = tf.multiply(alpha_attention_tile, attention_concat)
                print("c_attention shape is %s" % c_attention.get_shape())
                final_softmax_list.append(c_attention)
            final_softmax_list = tf.reshape(tf.convert_to_tensor(final_softmax_list), shape=[1, cfg.batch_size, -1, 1])
            print("final_softmax_list shape is %s" % final_softmax_list.get_shape())

            with tf.variable_scope("cnn_layer"):
                self.filter_layer = tf.get_variable(
                    'filter',
                    shape=(1, 2, 1, 2),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype=np.float32)
            psg_conv_layer = tf.nn.conv2d(final_softmax_list, self.filter_layer, strides=[1, 1, 1, 1], padding='SAME')
            print('psg_conv_layer shape is %s' % psg_conv_layer.get_shape())
            psg_relu_layer = tf.nn.relu(psg_conv_layer)
            psg_pool_layer = tf.layers.max_pooling2d(inputs=psg_relu_layer, pool_size=[1, 2], strides=[1, 2],
                                                     padding='valid')
            pos_pool_layer = tf.reshape(psg_pool_layer, shape=[cfg.batch_size, -1])
            print('pool_layer shape is %s' % pos_pool_layer.get_shape())

            with tf.variable_scope("proj_layer"):
                self.proj_layer = tf.get_variable(
                    'proj_layer',
                    shape=(cfg.context_window_size * cfg.target_lstm_hidden_size * 5, cfg.context_window_size),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            final_softmax = tf.nn.softmax(tf.matmul(pos_pool_layer, self.proj_layer), axis=1)
            print("final_softmax shape is %s" % final_softmax.get_shape())
            self.cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_prob, logits=final_softmax))
            print("cross_entropy_loss shape is %s" % self.cross_entropy_loss.get_shape())
            self.opt = tf.train.AdamOptimizer().minimize(self.cross_entropy_loss)
            self.model = tf.train.Saver()

            comparison = tf.equal(tf.argmax(final_softmax, axis=1), tf.argmax(self.validation_target_prob, axis=1))
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
        return self.sess.run([self.opt, self.cross_entropy_loss], feed_dict={
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
    if len(sys.argv) < 8:
        print("probabilistic_skip_gram_v3 <target> <context> <part-of-speech> <parser> "
              "<dictionary desc> <kb entity> <word emb output>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    total_sample_size = target.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_size_fake = int(total_batch_size * 1)

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    '''
    word_dictionary_size = 1000
    parser_dictionary_size = 50
    partofspeech_dictionary_size = 50
    kb_relation_dictionary_size = 500
    total_batch_size = 100
    word = np.random.randint(word_dictionary_size, size=(total_batch_size * cfg.batch_size))
    dictionary = np.random.randint(word_dictionary_size, size=(total_batch_size * cfg.batch_size, cfg.dict_time_step))
    kb_relation = np.random.randint(kb_relation_dictionary_size,
                                    size=(total_batch_size * cfg.batch_size, cfg.kb_relation_length))
    parser = np.random.randint(parser_dictionary_size,
                               size=(total_batch_size * cfg.batch_size, cfg.context_window_size))
    partofspeech = np.random.randint(partofspeech_dictionary_size, size=(total_batch_size * cfg.batch_size))
    context_info = np.random.randint(word_dictionary_size,
                                     size=(total_batch_size * cfg.batch_size, cfg.context_window_size))
    context_info_prob = np.random.rand(total_batch_size * cfg.batch_size, cfg.context_window_size)
    '''

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PSGModelObj = PSGModel(sess)
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.psg_test_summary_writer_path, sess.graph)

        trainable = False
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(total_batch_size):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                _, iter_loss = PSGModelObj.train(target[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 dict_desc[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 kb_entity[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 parser[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 pos[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 context[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                 context_info_prob[i*cfg.batch_size:(i+1)*cfg.batch_size])
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, np.sum(loss_sum) / cfg.batch_size / total_batch_size))
            train_loss = PSGModelObj.get_loss_summary(np.sum(loss_sum) / cfg.batch_size / total_batch_size)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for j in range(total_batch_size):
                iter_accuracy = PSGModelObj.validate(target[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     dict_desc[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     kb_entity[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     parser[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     pos[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                     context[i*cfg.batch_size:(i+1)*cfg.batch_size],
                                                    context_info_prob[i*cfg.batch_size:(i+1)*cfg.batch_size])
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