#!/bin/bash
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 w2v \
~/prob_skip_gram/log/w2v.model.sg.ptb cnn 3 1>log
echo "SICK coarse-grained skip-gram cnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 w2v \
~/prob_skip_gram/log/w2v.model.sg.ptb rnn 3 1>log
echo "SICK coarse-grained skip-gram rnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 w2v \
~/prob_skip_gram/log/w2v.model.cbow.ptb cnn 3 1>log
echo "SICK coarse-grained cbow cnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 w2v \
~/prob_skip_gram/log/w2v.model.cbow.ptb rnn 3 1>log
echo "SICK coarse-grained cbow rnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 glove \
~/GloVe/vectors.txt cnn 3 1>log
echo "SICK coarse-grained glove cnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 glove \
~/GloVe/vectors.txt rnn 3 1>log
echo "SICK coarse-grained glove rnn ptb Done."
python entailment_classification.py ~/prob_skip_gram/data/classification/SICK/data1 glove \
~/GloVe/vectors.txt rnn 3 1>log