#!/usr/bin/env python

"""
File that will load the trained weights of the core network, and train a second
encoder from a different datasource.
"""


from neon.backends import gen_backend
from neon.models import Model
from neon.layers.container import Sequential, MergeMultistream, MergeBroadcast
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Accuracy
from neon.util.argparser import NeonArgparser, extract_valid_args

from networks import load_core, get_title_augmentor
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
    train_content, test_content = get_saudinet_data(args, modality='content')
    train_title, test_title = get_saudinet_data(args, modality='title')

    # Build the network
    encoder, decoder = load_core()
    encoder = Sequential(encoder)
    # decoder = Sequential(decoder)
    augmentor, cost, opt = get_title_augmentor()
    augmentor = Sequential(augmentor)
    model = Model(layers=[
        # MergeMultistream(layers=[encoder, ], merge='stack'),
        MergeBroadcast(layers=[encoder, augmentor], merge='stack'),
        decoder
    ])

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

    # Save model
    # TODO: Export augmentor
