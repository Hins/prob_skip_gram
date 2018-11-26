# -*- coding: utf-8 -*-
# @Time        : 2018/9/3 18:10
# @Author      : panxiaotong
# @Description : configurations for PSG(probabilistic skip-gram) model

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('word_embedding_size', 256, 'word id embedding size')
flags.DEFINE_integer('dict_time_step', 15, 'dictionary sentence time step')
flags.DEFINE_integer('kb_embedding_size', 200, 'knowledge base relation embedding size')
flags.DEFINE_integer('kb_relation_length', 5, 'knowledge base relation one-hot length')
flags.DEFINE_integer('parser_embedding_size', 300, 'parser embedding size with another word')
flags.DEFINE_integer('partofspeech_embedding_size', 400, 'part of speech embedding size')
flags.DEFINE_integer('context_window_size', 4, 'context window size')
flags.DEFINE_integer('dict_lstm_hidden_size', 128, 'dictionary lstm model hidden size')
flags.DEFINE_integer('target_lstm_hidden_size', 300, 'target lstm model hidden size')
flags.DEFINE_integer('attention_size', 400, 'attention parameter v size')
flags.DEFINE_float('normalize_value', 1.33, 'positive sample normalized value')

flags.DEFINE_float('train_set_ratio', 0.8, 'train set ratio')
flags.DEFINE_integer('batch_size', 150, 'train batch size')
flags.DEFINE_integer('accumulative_metric_count', 5, 'accumulative metric count')

flags.DEFINE_float('stddev', 0.8, 'stddev for W initializer')
flags.DEFINE_integer('negative_sample_size', 5, 'negative sample size')
flags.DEFINE_integer('epoch_size', 100, 'epoch size')
flags.DEFINE_integer('early_stop_iter', 10, 'early stop iteration')

flags.DEFINE_integer('sentiment_classification_num', 5, 'sentiment analysis classification number')

flags.DEFINE_string('coocur_separator', '#', 'words coocurrence separator')

flags.DEFINE_string('', )

flags.DEFINE_string('summaries_dir', '../tb/PSG', 'Summaries directory')
flags.DEFINE_string('psg_train_summary_writer_path', '/psg_train', 'psg train summary writer path')
flags.DEFINE_string('psg_test_summary_writer_path', '/psg_test', 'psg test summary writer path')
flags.DEFINE_string('sg_train_summary_writer_path', '/sg_train', 'skip-gram train summary writer path')
flags.DEFINE_string('sg_test_summary_writer_path', '/sg_test', 'skip-gram test summary writer path')

cfg = tf.app.flags.FLAGS