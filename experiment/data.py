#!/usr/bin/env python
"""
File responsible for downloading the data and setting up the single and
multi-data stream.
"""
import sys, os, json, codecs, urllib, zipfile
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
                zf = zipfile.ZipFile(open(path ,"rb"))
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
            if "json" in path :
                fin = codecs.open(path, "r")
                for entry in json.load(fin):
                    article.append(entry)
    return article

