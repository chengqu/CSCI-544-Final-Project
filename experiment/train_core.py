#!/usr/bin/env python

"""
File that will train the core network, consisting of the initial encoder and
the decoder.
"""

from neon.backends import gen_backend
from neon.data import ArrayIterator
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Accuracy
from neon.layers.container import Sequential
from neon.util.argparser import NeonArgparser, extract_valid_args

from networks import get_core_net, save_core
from data import get_imdb, get_saudinet_data

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

# hyperparameters from the reference
args.batch_size = 128
args.backend = 'gpu'


if __name__ == '__main__':
    # Setup backend
    be = gen_backend(**extract_valid_args(args, gen_backend))

    # Load the dataset
    # train_set, test_set, nout = get_imdb(args)
    (X_train, y_train), (X_test, y_test), nout, vocab_size = get_saudinet_data(args, modality='content')
    train_set = ArrayIterator(X_train, y_train, nout)
    test_set = ArrayIterator(X_test, y_test, nout)

    # Build the network
    (encoder, decoder), cost, opt = get_core_net(nout=nout, vocab_size=vocab_size)
    model = Model(layers=Sequential([encoder, decoder, ]))

    callbacks = Callbacks(model, eval_set=test_set, **args.callback_args)

    # Train the model on the dataset
    model.fit(
        train_set,
        optimizer=opt,
        cost=cost,
        num_epochs=args.epochs,
        callbacks=callbacks)

    # Benchmark
    print 'Train accuracy: ', model.eval(train_set, metric=Accuracy())
    print 'Test accuracy: ', model.eval(test_set, metric=Accuracy())

    # Save models
    save_core(encoder, decoder)
