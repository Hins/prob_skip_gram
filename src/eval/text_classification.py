# -*- coding: utf-8 -*-
# @Time        : 2018/11/22 15:50
# @Author      : panxiaotong
# @Description : MIPSG-I/MIPSG-II model, text classification

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.contrib import rnn
from config.config import cfg
import numpy as np
import gensim

target_list = []
test_list = []
test_labels = []
emb_list = []
labels = []
word_dictionary_size = 0
def load_sample(input_file, test_file, word_emb_type, word_emb_file, label_type, num_classes):
    global target_list, test_list, test_labels, emb_list, labels, word_dictionary_size

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
            target_list.append(tmp_list)
            label_array = np.zeros(shape=[num_classes], dtype=np.int32)
            if label_type == 'int':
                label_array[int(elements[1])] = 1
            elif label_array == 'float':
                label_array[np.round(float(num_classes) * float(elements[1]))] = 1
            labels.append(label_array)
        f.close()

    if test_file.strip() != '':
        with open(test_file, 'r') as f:
            for line in f:
                elements = line.strip('\r\n').split('\t')
                tmp_list = elements[0].split(',')
                if len(tmp_list) > cfg.dict_time_step:
                    tmp_list = tmp_list[0:cfg.dict_time_step]
                else:
                    num = cfg.dict_time_step - len(tmp_list)
                    for i in range(num):
                        tmp_list.append("0")
                test_list.append(tmp_list)
                label_array = np.zeros(shape=[num_classes], dtype=np.int32)
                if label_type == 'int':
                    label_array[int(elements[1])] = 1
                elif label_array == 'float':
                    label_array[np.round(float(num_classes) * float(elements[1]))] = 1
                test_labels.append(label_array)
            f.close()

    if word_emb_type == 'file':
        with open(word_emb_file, 'r') as f:
            for idx, line in enumerate(f):
                emb_list.append([float(item) for item in line.strip('\r\n').split(',')])
            word_dictionary_size = idx + 1
            f.close()
    elif word_emb_type == 'w2v':
        model = gensim.models.Word2Vec.load(word_emb_file)
        word_dict = {}
        for k, v in model.wv.vocab.items():
            word_dict[k] = v.index
        sorted_list = sorted(word_dict, key=word_dict.get)
        for word in sorted_list:
            emb_list.append([float(item) for item in model[word]])
        word_dictionary_size = len(word_dict)
    elif word_emb_type == 'glove':
        word_dict = {}
        with open(word_emb_file, 'r') as f:
            for idx, line in enumerate(f):
                elements = line.strip('\r\n').split(' ')
                word_dict[elements[0]] = idx
                emb_list.append([float(item) for sub_idx, item in enumerate(elements) if sub_idx > 0])
            f.close()
        word_dictionary_size = len(word_dict)

    target_list = np.asarray(target_list, dtype=np.int32)
    test_list = np.asarray(test_list, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    emb_list = np.asarray(emb_list, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

class textClassificationModel():
    def __init__(self, sess, train_method, num_classes):
        self.word_emb = tf.placeholder(shape=[word_dictionary_size, cfg.word_embedding_size], dtype=tf.float32)
        self.sentence = tf.placeholder(shape=[cfg.batch_size, cfg.dict_time_step], dtype=tf.int32)
        self.label = tf.placeholder(shape=[cfg.batch_size, num_classes], dtype=tf.float32)
        self.sess = sess

        with tf.device('/gpu:1'):
            samples = tf.split(self.sentence, num_or_size_splits=cfg.batch_size, axis=0)
            emb_list = []
            for sample in samples:
                emb_list.append(tf.squeeze(tf.nn.embedding_lookup(self.word_emb, sample)))
            sampled = tf.convert_to_tensor(emb_list)
            if train_method == 'rnn':
                cell = rnn.BasicLSTMCell(cfg.dict_lstm_hidden_size, name="dict_rnn")
                init_state = cell.zero_state(cfg.batch_size, dtype=tf.float32)
                _, rnn_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=sampled,
                                                     initial_state=init_state, time_major=False)
                self.logit_w = tf.get_variable(
                    'logit_w',
                    shape=(cfg.dict_lstm_hidden_size, num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.logit_b = tf.get_variable(
                    'logit_b',
                    shape=(num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                x = tf.squeeze(tf.nn.bias_add(tf.matmul(rnn_final_state.h, self.logit_w), self.logit_b))
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=self.label))
                self.opt = tf.train.AdamOptimizer().minimize(self.loss)

                self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(x, 1), tf.argmax(self.label, 1)), dtype=tf.float32))
            elif train_method == 'cnn':
                self.filter_layer = tf.get_variable(
                    'filter',
                    shape=(1, 2, 1, 2),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype=np.float32)
                conv_layer = tf.nn.conv2d(tf.reshape(tf.reduce_sum(sampled, axis=1), shape=[1, cfg.batch_size, -1, 1]),
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
                    shape=(pool_layer.get_shape()[1], num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.logit_b = tf.get_variable(
                    'logit_b',
                    shape=(num_classes),
                    initializer=tf.truncated_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

                x = tf.squeeze(tf.nn.bias_add(tf.matmul(pool_layer, self.logit_w), self.logit_b))
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=self.label))
                self.opt = tf.train.AdamOptimizer().minimize(self.loss)

                self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(x, 1), tf.argmax(self.label, 1)), dtype=tf.float32))

            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word_emb, sentence, label):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word_emb: word_emb,
            self.sentence: sentence,
            self.label: label
        })

    def validate(self, word_emb, sentence, label):
        return self.sess.run([self.accuracy], feed_dict={
            self.word_emb: word_emb,
            self.sentence: sentence,
            self.label: label
        })

if __name__ == '__main__':
    if len(sys.argv) < 8:
        print("text_classification <input> <test_set> <word_emb_type> <word emb> <label type> <train_method> <num_classes>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2], sys.argv[3].lower(), sys.argv[4], sys.argv[5].lower(), int(sys.argv[7]))
    total_sample_size = target_list.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_offset = int(total_sample_size * cfg.train_set_ratio)
    if len(test_list) != 0:
        train_set_size = total_batch_size
    test_batch_size = test_list.shape[0] / cfg.batch_size

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sentimentObj = textClassificationModel(sess, sys.argv[6].lower(), int(sys.argv[7]))
        tf.global_variables_initializer().run()

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print(k)

        trainable = False
        max_accuracy = -1.0
        prev_avg_accu = 0.0
        cur_avg_accu = 0.0
        prev_loss = 0.0
        cur_loss = 0.0
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(train_set_size):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                _, iter_loss = sentimentObj.train(emb_list,
                                                  target_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                                  labels[i * cfg.batch_size:(i+1) * cfg.batch_size])
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, loss_sum / cfg.batch_size))

            accuracy = 0.0
            if len(test_list) == 0:
                for j in range(total_batch_size - train_set_size):
                    k = j + train_set_size
                    iter_accuracy = sentimentObj.validate(emb_list,
                                                          target_list[k * cfg.batch_size:(k + 1) * cfg.batch_size],
                                                          labels[k * cfg.batch_size:(k + 1) * cfg.batch_size])
                    accuracy += iter_accuracy[0]
                accuracy /= (total_batch_size - train_set_size)
                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                print("epoch_index %d : accuracy %f" % (epoch_index, accuracy))

            else:
                for j in range(test_batch_size):
                    iter_accuracy = sentimentObj.validate(emb_list,
                                                          test_list[j * cfg.batch_size:(j + 1) * cfg.batch_size],
                                                          test_labels[j * cfg.batch_size:(j + 1) * cfg.batch_size])
                    accuracy += iter_accuracy[0]
                accuracy /= test_batch_size
                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                print("epoch_index %d : accuracy %f" % (epoch_index, accuracy))

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

        print('max_accuracy is %f' % float("{0:.3f}".format(max_accuracy)))
        sess.close()