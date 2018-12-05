# -*- coding: utf-8 -*-
# @Time        : 2018/8/28 10:56
# @Author      : panxiaotong
# @Description : naive skip-gram for comparison

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append("..")
import tensorflow as tf
from config.config import cfg
import numpy as np

target_list = []
context_list = []
labels = []
new_context_list = []
word_dictionary_size = 0
def load_sample(input_file, dict_file, context_file):
    global target_list, context_list, labels, new_context_list, word_dictionary_size
    word_dict = {}
    with open(dict_file, 'r') as f:
        for line in f:
            line = line.strip('\r\n').split('\t')
            word_dict[line[0]] = int(line[1])
        f.close()
    word_dictionary_size = len(word_dict)

    with open(input_file, 'r') as f:
        for line in f:
            target_list.append(line.strip('\r\n'))
        f.close()

    with open(context_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(',')
            context_list.extend(elements)
        f.close()
    target_list = np.asarray(target_list)
    context_list = np.asarray(context_list)
    labels = np.zeros(shape=[context_list.shape[0], cfg.negative_sample_size + 1], dtype=np.int32)
    new_context_list = np.zeros(shape=[context_list.shape[0]], dtype=np.int32)

class SkipGramModel():
    def __init__(self, sess, output_file):
        self.word = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.target = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.validation_index = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.validation_target = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.sess = sess
        self.output_file = output_file

        with tf.device('/gpu:1'):
            with tf.variable_scope("skipgram_model"):
                self.word_embed_weight = tf.get_variable(
                    'word_emb',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                    dtype='float32'
                )

            # [cfg.batch_size, cfg.word_embedding_size]
            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word)
            print('word_embed_init shape is %s' % word_embed_init.get_shape())
            tanh_word_embed_init = tf.nn.tanh(word_embed_init)
            # [cfg.batch_size, cfg.word_embedding_size]
            print('tanh_word_embed_init shape is %s' % tanh_word_embed_init.get_shape())

            with tf.variable_scope("skipgram_model"):
                proj_weight = tf.get_variable(
                    'proj_layer',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                    dtype='float32'
                )

                print("proj_weight shape is %s" % proj_weight.get_shape())
                proj_bias = tf.get_variable(
                    'proj_bias',
                    shape=(word_dictionary_size),
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                    dtype='float32'
                )
                print("proj_bias shape is %s" % proj_bias.get_shape())
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=proj_weight, biases=proj_bias, labels=tf.reshape(self.target, shape=[-1,1]),
                               inputs=tanh_word_embed_init, num_sampled=cfg.negative_sample_size, num_classes=word_dictionary_size))
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            self.model = tf.train.Saver()

            proj_layer = tf.nn.bias_add(tf.reshape(
                tf.matmul(proj_weight, tf.reshape(tanh_word_embed_init, shape=[cfg.word_embedding_size, -1])),
                shape=[-1, word_dictionary_size]
            ), proj_bias)
            # [cfg.batch_size, word_dictionary_size]
            print("proj_layer shape is %s" % proj_layer.get_shape())

            # validation:
            softmax_layer = tf.reshape(tf.nn.softmax(logits=proj_layer, axis=1), shape=[cfg.batch_size, -1])
            print("softmax_layer shape is %s" % softmax_layer.get_shape())
            softmax_layer_list = tf.split(softmax_layer, num_or_size_splits=cfg.batch_size)
            validation_index_list = tf.split(self.validation_index, num_or_size_splits=cfg.batch_size)
            embed_list = []
            for layer_index, layer in enumerate(softmax_layer_list):
                emb_result = tf.squeeze(
                    tf.nn.embedding_lookup(tf.reshape(layer, shape=[-1, 1]), validation_index_list[layer_index]))
                embed_list.append(emb_result)
            index_score = tf.convert_to_tensor(embed_list)
            print("index_score shape is %s" % index_score.get_shape())
            self.predict_result = tf.cast(tf.argmax(index_score, axis=1), dtype=tf.int32)
            print("predict_result shape is %s" % self.predict_result.get_shape())

            comparison = tf.equal(self.validation_target, self.predict_result)
            print("comparison shape is %s" % comparison.get_shape())
            self.accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))

            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word1, word2):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word: word1,
            self.target: word2})

    def validate(self, word1, validation_index, validation_target):
        return self.sess.run([self.predict_result, self.accuracy], feed_dict={
            self.word: word1,
            self.validation_index: validation_index,
            self.validation_target: validation_target
        })

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

    def get_word_emb(self):
        return self.sess.run(self.word_embed_weight, feed_dict={})

    def save(self):
        self.model.save(sess, self.output_file)

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("skipgram <input file> <dict file> <context file> <negative sample file> <output model> <word emb model>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2], sys.argv[3])
    total_sample_size = context_list.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_offset = int(total_sample_size * cfg.train_set_ratio)
    train_set_size_fake = int(total_batch_size * 1)

    targets = []
    for i in target_list:
        for j in range(cfg.context_window_size):
            targets.append(i)
    targets = np.asarray(targets)

    with open(sys.argv[4], 'r') as f:
        for idx, line in enumerate(f):
            elements = [int(item) for item in line.strip('\r\n').split(',')]
            labels[idx] = np.asarray(elements)
            for sub_idx, element in enumerate(elements):
                if int(element) == int(context_list[idx]):
                    new_context_list[idx] = sub_idx
                    break
        f.close()

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        SkipGramObj = SkipGramModel(sess, sys.argv[5])
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.sg_train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.sg_test_summary_writer_path, sess.graph)

        trainable = False
        prev_avg_accu = 0.0
        cur_avg_accu = 0.0
        prev_loss = 0.0
        cur_loss = 0.0
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(train_set_size_fake):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                _, iter_loss = SkipGramObj.train(targets[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                                 context_list[i * cfg.batch_size:(i+1) * cfg.batch_size])
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, loss_sum / cfg.batch_size))
            train_loss = SkipGramObj.get_loss_summary(loss_sum / cfg.batch_size)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for j in range(total_batch_size - train_set_size):
                k = j + train_set_size
                predict_result, iter_accuracy = SkipGramObj.validate(targets[k*cfg.batch_size : (k+1)*cfg.batch_size],
                                                     labels[k*cfg.batch_size : (k+1)*cfg.batch_size],
                                                     new_context_list[k*cfg.batch_size : (k+1)*cfg.batch_size])
                accuracy += iter_accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
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
            test_accuracy = SkipGramObj.get_accuracy_summary(accuracy / (total_batch_size - train_set_size))
            test_writer.add_summary(test_accuracy, epoch_index + 1)

        embed_weight = SkipGramObj.get_word_emb()
        output_embed_file = open(sys.argv[6], 'w')
        for embed_item in embed_weight:
            embed_list = list(embed_item)
            embed_list = [str(item) for item in embed_list]
            output_embed_file.write(','.join(embed_list) + '\n')
        output_embed_file.close()
        SkipGramObj.save()
        sess.close()