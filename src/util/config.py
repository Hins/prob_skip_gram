# -*- coding: utf-8 -*-
# @Time        : 2018/9/3 18:10
# @Author      : panxiaotong
# @Description : configurations for PSG(probabilistic skip-gram) model

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('word_embedding_size', 256, 'word id embedding size')
flags.DEFINE_integer('dict_embedding_size', 128, 'dictionary explanation embedding size')
flags.DEFINE_integer('dict_length', 15, 'dictionary length')
flags.DEFINE_integer('kb_embedding_size', 256, 'knowledge base relation embedding size')
flags.DEFINE_integer('kb_relation_length', 5, 'knowledge base relation one-hot length')
flags.DEFINE_integer('parser_embedding_size', 128, 'parser embedding size with another word')
flags.DEFINE_integer('partofspeech_embedding_size', 128, 'part of speech embedding size')
flags.DEFINE_integer('context_window_size', 4, 'context window size')
flags.DEFINE_integer('hidden_size', 128, 'word2vec weight size')

flags.DEFINE_float('train_set_ratio', 0.8, 'train set ratio')
flags.DEFINE_integer('batch_size', 256, 'train batch size')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_integer('negative_sample_size', 5, 'negative sample size')
flags.DEFINE_integer('epoch_size', 10, 'epoch size')

flags.DEFINE_string('summaries_dir', '../tb/PSG', 'Summaries directory')
flags.DEFINE_string('psg_train_summary_writer_path', '/psg_train', 'psg train summary writer path')
flags.DEFINE_string('psg_test_summary_writer_path', '/psg_test', 'psg test summary writer path')
flags.DEFINE_string('sg_train_summary_writer_path', '/sg_train', 'skip-gram train summary writer path')
flags.DEFINE_string('sg_test_summary_writer_path', '/sg_test', 'skip-gram test summary writer path')

cfg = tf.app.flags.FLAGS