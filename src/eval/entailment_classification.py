# -*- coding: utf-8 -*-
# @Time        : 2018/11/22 15:50
# @Author      : panxiaotong
# @Description : MIPSG-I/MIPSG-II model, address entailment classification problems

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.contrib import rnn
from util.config import cfg
import numpy as np

sent1_list = []
sent2_list = []
emb_list = []
labels = []
label_category_list = []
sample_list = []
word_dictionary_size = 0
def load_sample(input_file, word_emb_file):
    global sent1_list, sent2_list, emb_list, labels, word_dictionary_size

    with open(input_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            tmp_list = elements[0].split(',')
            if len(tmp_list) > cfg.dict_time_step:
                tmp_list = tmp_list[0:cfg.dict_time_step]
            else:
                num = cfg.dict_time_step - len(tmp_list)
                for i in range(num):
                    tmp_list.append("0")
            sent1_list.append(tmp_list)
            tmp_list = elements[1].split(',')
            if len(tmp_list) > cfg.dict_time_step:
                tmp_list = tmp_list[0:cfg.dict_time_step]
            else:
                num = cfg.dict_time_step - len(tmp_list)
                for i in range(num):
                    tmp_list.append("0")
            sent2_list.append(tmp_list)
            label = elements[3]
            labels.append(label)
            if label not in label_category_list:
                label_category_list.append(label)
        f.close()

    with open(word_emb_file, 'r') as f:
        for idx, line in enumerate(f):
            emb_list.append([float(item) for item in line.strip('\r\n').split(',')])
        word_dictionary_size = idx + 1
        f.close()

    sent1_list = np.asarray(sent1_list, dtype=np.int32)
    sent2_list = np.asarray(sent2_list, dtype=np.int32)
    emb_list = np.asarray(emb_list, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

class entailmentModel():
    def __init__(self, sess, train_method):
        self.word_emb = tf.placeholder(shape=[word_dictionary_size, cfg.word_embedding_size], dtype=tf.float32)
        self.sent1 = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.sent2 = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.label = tf.placeholder(shape=[cfg.batch_size], dtype=tf.float32)
        self.sess = sess

        with tf.device('/gpu:1'):
            sent1 = tf.split(self.sent1, num_or_size_splits=cfg.batch_size, axis=0)
            sent2 = tf.split(self.sent2, num_or_size_splits=cfg.batch_size, axis=0)
            emb1_list = []
            emb2_list = []
            for idx, sample in enumerate(sent1):
                emb1_list.append(tf.squeeze(tf.nn.embedding_lookup(self.word_emb, sample)))
                emb2_list.append(tf.squeeze(tf.nn.embedding_lookup(self.word_emb, sent2[idx])))
            emb1_list = tf.convert_to_tensor(emb1_list)
            print("emb1_list shape is %s" % emb1_list.get_shape())
            emb2_list = tf.convert_to_tensor(emb2_list)
            print("emb2_list shape is %s" % emb2_list.get_shape())
            if train_method == 'rnn':
                cell = rnn.BasicLSTMCell(cfg.dict_lstm_hidden_size)
                init_state = cell.zero_state(cfg.batch_size, dtype=tf.float32)
                with tf.variable_scope("sent1"):
                    _, rnn_final_state1 = tf.nn.dynamic_rnn(cell=cell, inputs=emb1_list,
                                                         initial_state=init_state, time_major=False)
                cell = rnn.BasicLSTMCell(cfg.dict_lstm_hidden_size)
                init_state = cell.zero_state(cfg.batch_size, dtype=tf.float32)
                with tf.variable_scope("sent2"):
                    _, rnn_final_state2 = tf.nn.dynamic_rnn(cell=cell, inputs=emb2_list,
                                                        initial_state=init_state, time_major=False)
                lstm_state = tf.concat([rnn_final_state1.h, rnn_final_state2.h], axis=1)
                print("lstm_state shape is %s" % lstm_state.get_shape())
                self.logit_w = tf.get_variable(
                    'logit_w',
                    shape=(cfg.dict_lstm_hidden_size * 2, 1),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.logit_b = tf.get_variable(
                    'logit_b',
                    shape=(1),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                x = tf.squeeze(tf.nn.bias_add(tf.matmul(lstm_state, self.logit_w), self.logit_b))
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.label))
                self.opt = tf.train.AdamOptimizer().minimize(self.loss)

                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(tf.nn.sigmoid(x)), self.label), tf.float32))
            elif train_method == 'cnn':
                sampled = tf.concat([tf.reduce_sum(emb1_list, axis=1),
                                     tf.reduce_sum(emb2_list, axis=1)], axis=1)
                print("sampled shape is %s" % sampled.get_shape())
                self.filter_layer = tf.get_variable(
                    'filter',
                    shape=(1, 2, 1, 2),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype=np.float32)
                conv_layer = tf.nn.conv2d(tf.reshape(sampled, shape=[1, cfg.batch_size, -1, 1]),
                                          self.filter_layer, strides=[1, 1, 1, 1],
                                          padding='SAME')
                print('conv_layer shape is %s' % conv_layer.get_shape())
                relu_layer = tf.nn.relu(conv_layer)
                pool_layer = tf.layers.max_pooling2d(inputs=relu_layer, pool_size=[1, 2], strides=[1, 2],
                                                         padding='valid')
                pool_layer = tf.reshape(pool_layer, shape=[cfg.batch_size, -1])
                print('pool_layer shape is %s' % pool_layer.get_shape())

                self.logit_w = tf.get_variable(
                    'logit_w',
                    shape=(pool_layer.get_shape()[1], 1),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.logit_b = tf.get_variable(
                    'logit_b',
                    shape=(1),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

                x = tf.squeeze(tf.nn.bias_add(tf.matmul(pool_layer, self.logit_w), self.logit_b))
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.label))
                self.opt = tf.train.AdamOptimizer().minimize(self.loss)
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(tf.nn.sigmoid(x)), self.label), tf.float32))

            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word_emb, sent1, sent2, label):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word_emb: word_emb,
            self.sent1: sent1,
            self.sent2: sent2,
            self.label: label
        })

    def validate(self, word_emb, sent1, sent2, label):
        return self.sess.run(self.accuracy, feed_dict={
            self.word_emb: word_emb,
            self.sent1: sent1,
            self.sent2: sent2,
            self.label: label
        })

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("skipgram <input> <word emb> <train_method>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2])
    total_sample_size = sent1_list.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_offset = int(total_sample_size * cfg.train_set_ratio)
    train_set_size_fake = int(total_batch_size * 1)

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sentimentObj = entailmentModel(sess, sys.argv[3].lower())
        tf.global_variables_initializer().run()

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print(k)

        max_accuracy = -1.0
        trainable = False
        for pos_label in label_category_list:
            pos_label = int(pos_label)
            new_label = np.zeros(shape=[total_sample_size], dtype=np.float32)
            row = labels.shape[0]
            for idx in range(row):
                if pos_label == labels[idx]:
                    new_label[idx] = 1
            for epoch_index in range(cfg.epoch_size):
                loss_sum = 0.0
                for i in range(train_set_size_fake):
                    if trainable is True:
                        tf.get_variable_scope().reuse_variables()
                    trainable = True
                    _, iter_loss = sentimentObj.train(emb_list,
                                                      sent1_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                                      sent2_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                                      new_label[i * cfg.batch_size:(i+1) * cfg.batch_size])
                    loss_sum += iter_loss
                print("epoch_index %d, loss is %f" % (epoch_index, loss_sum / cfg.batch_size))

                accuracy = 0.0
                for j in range(total_batch_size - train_set_size):
                    k = j + train_set_size
                    item_accuracy = sentimentObj.validate(emb_list,
                                                           sent1_list[k * cfg.batch_size:(k + 1) * cfg.batch_size],
                                                           sent2_list[k * cfg.batch_size:(k + 1) * cfg.batch_size],
                                                           new_label[k * cfg.batch_size:(k + 1) * cfg.batch_size])
                    accuracy += item_accuracy
                    if item_accuracy > max_accuracy:
                        max_accuracy = item_accuracy
                print("epoch_index %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
        sess.close()