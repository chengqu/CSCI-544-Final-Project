#!/usr/bin/env python

"""
File that will train the core network, consisting of the initial encoder and
the decoder.
"""

from neon.backends import gen_backend
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Accuracy
from neon.util.argparser import NeonArgparser, extract_valid_args

from networks import get_core_net
from data import get_imdb

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
    # TODO: Change IMDB with Arabic articles
    train_set, test_set = get_imdb(args)

    # Build the network
    (encoder, decoder), cost, opt = get_core_net(nout=2)
    from neon.layers.container import Sequential
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
    path = './saved_models/'
    Model(encoder).save_params(path + 'encoder.neon', True)
    Model(decoder).save_params(path + 'decoder.neon', True)
