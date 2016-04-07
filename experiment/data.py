#!/usr/bin/env python

"""
File responsible for downloading the data and setting up the single and
multi-data stream.
"""

from neon.data import ArrayIterator
from neon.data.dataloaders import load_imdb
from neon.data.text_preprocessing import pad_data

VOCAB_SIZE = 20000
SENTENCE_LENGTH = 128


def get_imdb(args):
    path = load_imdb(path=args.data_dir)
    (X_train, y_train), (X_test, y_test), nclass = pad_data(
        path, vocab_size=VOCAB_SIZE, sentence_length=SENTENCE_LENGTH)
    train_set = ArrayIterator(X_train, y_train, nclass=2)
    test_set = ArrayIterator(X_test, y_test, nclass=2)
    return (train_set, test_set)
