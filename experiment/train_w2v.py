#!/usr/bin/env python

"""
File that will train the word2vec model
"""


from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.initializers import Uniform, GlorotUniform, Array
from neon.layers import GeneralizedCost, Affine, Dropout, LookupTable, LSTM, RecurrentSum
from neon.models import Model
from neon.optimizers import Adagrad
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.callbacks.callbacks import Callbacks
from neon.data.text_preprocessing import get_paddedXY, get_google_word2vec_W
import h5py
import cPickle
from data import load_articles, preprocess_data_train
import os

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-f', '--review_file',
                    default='labeledTrainData.tsv',
                    help='input movie review file')
parser.add_argument('--vocab_file',
                    default='labeledTrainData.tsv.vocab',
                    help='output file to save the processed vocabulary')
parser.add_argument('--use_w2v', action='store_true',
                    help='use downloaded Google Word2Vec')
parser.add_argument('--w2v',
                    default='GoogleNews-vectors-negative300.bin',
                    help='the pre-built Word2Vec')
args = parser.parse_args()
args.backend = 'gpu'



# hyperparameters
hidden_size = 128
embedding_dim = 128
vocab_size = 343172
sentence_length = 128
batch_size = 32
gradient_limit = 5
clip_gradients = True
num_epochs = args.epochs
embedding_update = True



#if __name__ == '__main__':

# Setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# get the preprocessed and tokenized data
article = load_articles("../SaudiNewsNet")
fname_h5, fname_vocab = preprocess_data_train(article, 'content')


w2v_file = "GoogleNews-vectors-negative300.bin"

if not os.path.exists(w2v_file):
    print "Please download GoogleNews-vectors-negative300.bin from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing  and put it in current directory"




print "Start to load the Word2Vec vectors: embedding size - {}".format(embedding_dim)
vocab, rev_vocab = cPickle.load(open(fname_vocab, 'rb'))
init_emb_np, embedding_dim, _ = get_google_word2vec_W(w2v_file, vocab,vocab_size=vocab_size, index_from=3)
init_emb = Array(val=be.array(init_emb_np))

print "Done loading the Word2Vec vectors: embedding size - {}".format(embedding_dim)
embedding_update = True

print "Loading training and validation data ..."

h5f = h5py.File(fname_h5, 'r')
data, h5train, h5valid = h5f['data'], h5f['train'], h5f['valid']
ntrain, nvalid, nclass, vocab_size = data.attrs['ntrain'], data.attrs['nvalid'], data.attrs['nclass'], data.attrs['vocab_size']


# make train dataset
Xy = h5train[:ntrain]
X = [xy[1:] for xy in Xy]
y = [xy[0] for xy in Xy]
X_train, y_train = get_paddedXY(
                                X, y, vocab_size=vocab_size, sentence_length=sentence_length)
train_set = ArrayIterator(X_train, y_train, nclass=nclass)

# make valid dataset
Xy = h5valid[:nvalid]
X = [xy[1:] for xy in Xy]
y = [xy[0] for xy in Xy]
X_valid, y_valid = get_paddedXY(
                                X, y, vocab_size=vocab_size, sentence_length=sentence_length)
valid_set = ArrayIterator(X_valid, y_valid, nclass=nclass)


print "Initialize model ..."



# initialization
init_glorot = GlorotUniform()


# define layers
layers = [
          LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb,
                      pad_idx=0, update=embedding_update),
          LSTM(hidden_size, init_glorot, activation=Tanh(), gate_activation=Logistic(),
               reset_cells=True),
          RecurrentSum(),
          Dropout(keep=0.5),
          Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
          ]

# set the cost, metrics, optimizer
cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
metric = Accuracy()
model = Model(layers=layers)
optimizer = Adagrad(learning_rate=0.01)

# configure callbacks
callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)

print "Start to train ..."

# train model
model.fit(train_set,
          optimizer=optimizer,
          num_epochs=num_epochs,
          cost=cost,
          callbacks=callbacks)

print "Done training"

# eval model
print "\nTrain Accuracy -", 100 * model.eval(train_set, metric=metric)
print "Test Accuracy -", 100 * model.eval(valid_set, metric=metric)









