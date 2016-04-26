#!/usr/bin/env python
"""
File responsible for downloading the data and setting up the single and
multi-data stream.
"""
import sys
import os
import json
import codecs
import urllib
import zipfile
import h5py
from collections import defaultdict
import regex
import numpy as np
import cPickle
from neon.data.text_preprocessing import get_paddedXY
from neon.data import ArrayIterator
from neon.data.dataloaders import load_imdb
from neon.data.text_preprocessing import pad_data

VOCAB_SIZE = 20000
SENTENCE_LENGTH = 128


def get_saudinet_data(args):
    """ Returns a train and validation dataset from SaudiNewsNet. """
    # get the preprocessed and tokenized data
    article = load_articles("./SaudiNewsNet")
    fname_h5, fname_vocab = preprocess_data_train(article, 'content', './experiments/labeledTrainData.tsv')

    h5f = h5py.File(fname_h5, 'r')
    data, h5train, h5valid = h5f['data'], h5f['train'], h5f['valid']
    ntrain, nvalid, nclass, vocab_size = data.attrs[
        'ntrain'], data.attrs['nvalid'], data.attrs['nclass'], data.attrs['vocab_size']

    # make train dataset
    Xy = h5train[:ntrain]
    X = [xy[1:] for xy in Xy]
    y = [xy[0] for xy in Xy]
    X_train, y_train = get_paddedXY(X, y, vocab_size=vocab_size, sentence_length=SENTENCE_LENGTH)
    train_set = ArrayIterator(X_train, y_train, nclass=nclass)

    # make valid dataset
    Xy = h5valid[:nvalid]
    X = [xy[1:] for xy in Xy]
    y = [xy[0] for xy in Xy]
    X_valid, y_valid = get_paddedXY(
        X, y, vocab_size=vocab_size, sentence_length=SENTENCE_LENGTH)
    valid_set = ArrayIterator(X_valid, y_valid, nclass=nclass)

    return train_set, valid_set, nclass


def get_imdb(args):
    """DUmmy dataset for dev."""
    path = load_imdb(path=args.data_dir)
    (X_train, y_train), (X_test, y_test), nclass = pad_data(
        path, vocab_size=VOCAB_SIZE, sentence_length=SENTENCE_LENGTH)
    train_set = ArrayIterator(X_train, y_train, nclass=2)
    test_set = ArrayIterator(X_test, y_test, nclass=2)
    return (train_set, test_set, 2)


"""
downloads the SaudiNewsNet and unzip dataset and returns the folder path
"""


def download_arabic_articles():
    url = "https://codeload.github.com/ParallelMazen/SaudiNewsNet/zip/master"
    zipfilename = "../SaudiNewsNet.zip"
    folder = zipfilename.replace(".zip", "")
    jsonfolder = os.path.join(folder, "json")
    if not os.path.exists(jsonfolder):
        os.makedirs(jsonfolder)

    saudi = urllib.URLopener()
    if not os.path.exists(zipfilename):
        saudi.retrieve(url, zipfilename)

    file = open(zipfilename, "rb")
    z = zipfile.ZipFile(file)
    z.extractall(folder)
    for subdir, dirs, files in os.walk("../SaudiNewsNet"):
        for f in files:
            path = os.path.join(subdir, f)
            if ".zip" in path:
                print path
                zf = zipfile.ZipFile(open(path, "rb"))
                zf.extractall(jsonfolder)
                zf.close()
    z.close()
    return jsonfolder


"""
returns a list of all entries in the dataset
"""


def load_articles(path):
    article = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(subdir, file)
            if "json" in path:
                fin = codecs.open(path, "r")
                for entry in json.load(fin):
                    article.append(entry)
    return article


def clean_string_unicode(string):
    string = regex.sub(ur"[^\P{P}(),!?\'\`]+", " ", string)
    string = regex.sub(ur",", " , ", string)
    string = regex.sub(ur"!", " ! ", string)
    string = regex.sub(ur"\(", " \( ", string)
    string = regex.sub(ur"\)", " \) ", string)
    string = regex.sub(ur"\?", " \? ", string)
    # string = regex.sub(ur"\s{2,}", " ", string)
    return string.strip()


def preprocess_data_train(data, query, path='.', filepath='labeledTrainData.tsv', vocab_file=None, vocab=None, train_ratio=0.8):

    fname_h5 = filepath + '.h5'
    if vocab_file is None:
        fname_vocab = filepath + '.vocab'
    else:
        fname_vocab = vocab_file

    if not os.path.exists(fname_h5) or not os.path.exists(fname_vocab):
        h5f = h5py.File(fname_h5, 'w')
        shape, maxshape = (2 ** 16,), (None, )
        dt = np.dtype([('y', np.uint8),
                       ('split', np.bool),
                       ('num_words', np.uint16),
                       ('text', h5py.special_dtype(
                        vlen=unicode))
                       ])
        data_text = h5f.create_dataset('data', shape=shape, maxshape=maxshape,
                                       dtype=dt, compression='gzip')
        data_train = h5f.create_dataset(
            'train', shape=shape, maxshape=maxshape, dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

        data_valid = h5f.create_dataset(
            'valid', shape=shape, maxshape=maxshape, dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')

        wdata = np.zeros((1, ), dtype=dt)
        labels = {}
        label_cnt = 0

        build_vocab = False
        if vocab is None:
            vocab = defaultdict(int)
            build_vocab = True
        nsamples = 0

        # possible keys
        # ['source','url','date_extracted','title','author','content']

        # Crawl through articles
        for i in range(0, len(data)):
            # Crawl through keys
            for key in data[i]:
                if key == 'source':  # Don't include labels (source) in vocabulary
                    continue
                data_ = clean_string_unicode(data[i][key])
                data_words = data_.strip().split()
                num_words = len(data_words)
                split = int(np.random.rand() < train_ratio)

                # create record
                if query == key:
                    lab = data[i]['source']
                    if lab not in labels:
                        labels[lab] = label_cnt
                        label_cnt += 1
                    wdata['y'] = labels[lab]
                    wdata['text'] = data_
                    wdata['num_words'] = num_words
                    wdata['split'] = split
                    data_text[i] = wdata

                # update the vocab
                # Vocab has to be built on all keys (all data)
                if build_vocab:
                    for word in data_words:
                        vocab[word] += 1

            nsamples += 1

        # histogram of class labels, sentence length
        ratings, counts = np.unique(
            data_text['y'][:nsamples], return_counts=True)
        sen_len, sen_len_counts = np.unique(
            data_text['num_words'][:nsamples], return_counts=True)
        vocab_size = len(vocab)
        nclass = len(ratings)
        data_text.attrs['vocab_size'] = vocab_size
        data_text.attrs['nrows'] = nsamples
        data_text.attrs['nclass'] = nclass
        data_text.attrs['class_distribution'] = counts
        print 'vocabulary size - ', vocab_size
        print '# of samples - ', nsamples
        print '# of classes', nclass
        print 'class distribution - ', ratings, counts
        sen_counts = zip(sen_len, sen_len_counts)
        sen_counts = sorted(sen_counts, key=lambda kv: kv[1], reverse=True)
        print 'sentence length - ', len(sen_len), sen_len, sen_len_counts

        if build_vocab:
            vocab_sorted = sorted(
                vocab.items(), key=lambda kv: kv[1], reverse=True)
            vocab = {}
            for i, t in enumerate(zip(*vocab_sorted)[0]):
                vocab[t] = i

        # map text to integers
        ntrain = 0
        nvalid = 0
        for i in range(nsamples):
            text = data_text[i]['text']
            y = data_text[i]['y']
            split = data_text[i]['split']
            text_int = [y] + [vocab[t] for t in text.strip().split()]
            if split:
                data_train[ntrain] = text_int
                ntrain += 1
            else:
                data_valid[nvalid] = text_int
                nvalid += 1
        data_text.attrs['ntrain'] = ntrain
        data_text.attrs['nvalid'] = nvalid
        print "# of train - {0}, # of valid - {1}".format(data_text.attrs['ntrain'],
                                                          data_text.attrs['nvalid'])
        # close open files
        h5f.close()

    if not os.path.exists(fname_vocab):
        rev_vocab = {}
        for wrd, wrd_id in vocab.iteritems():
            rev_vocab[wrd_id] = wrd
        print "vocabulary from dataset is saved into {}". format(fname_vocab)
        cPickle.dump((vocab, rev_vocab), open(fname_vocab, 'wb'))

    return fname_h5, fname_vocab


if __name__ == '__main__':
    article = load_articles("../SaudiNewsNet")
    fname_h5, fname_vocab = preprocess_data_train(article, 'content')
