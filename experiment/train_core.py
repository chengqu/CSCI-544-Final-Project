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

import argparse



if __name__ == '__main__':
    #parse parameter settings from command line
    parser = NeonArgparser(__doc__)
    parser.add_argument("-ds","--data_source", help="choose the data source to train initial encoder-decoder. defaul is content")
    parser.add_argument("-lr", "--learning_rate", type = float, help="set the learning rate. defaul 0.01")
    parser.add_argument("-es", "--embedding_size", type = int, help = "set the embedding size. default 128")
    parser.add_argument("-dr", "--dropout_rate", type = float, help = "set the dropout rate. default 0.5")
    parser.add_argument("-af", "--activation_function", type = int, help = "set the activation default: Tanh; 1: Tanh, 2: ReLU, 3: Sigmoid, 4: Normalizer")
    
    args = parser.parse_args(gen_be=False)

    # hyperparameters from the reference
    #args.batch_size = 128
    args.batch_size = 32 
    args.backend = 'gpu'
   

    if args.data_source:
        data_source = args.data_source
    else:
        args.data_source = 'content' 
    
    if args.learning_rate:
        learning_rate = args.learning_rate
    else:
        learning_rate = 0.01
    
    if args.embedding_size:
        embedding_size = args.embedding_size
    else:
        embedding_size = 128
    
    if args.dropout_rate:
        dropout_rate = args.dropout_rate
    else:
        dropout_rate = 0.5
    
    if args.activation_function:
        activation_function = args.activation_function
    else:
        activation_function = 1

    af={}
    af[1] = 'Tanh'
    af[2] = 'ReLU'
    af[3] = 'Sigmoid'
    af[4] = 'Normalizer'

    print ' '
    print 'data source used for initial encoder decoder is: ', args.data_source
    print 'training epochs is: ', args.epochs
    print 'learning rate is: ', learning_rate
    print 'embedding size is: ', embedding_size
    print 'dropout rate is: ', dropout_rate
    print 'activation functio is: ', af[activation_function]
    print ' '

    # Setup backend
    be = gen_backend(**extract_valid_args(args, gen_backend))

    # Load the dataset
    # train_set, test_set, nout = get_imdb(args)
    #(X_train, y_train), (X_test, y_test), nout, vocab_size = get_saudinet_data(args, modality='content')

    (X_train, y_train), (X_test, y_test), nout, vocab_size = get_saudinet_data(args, modality=args.data_source)


    train_set = ArrayIterator(X_train, y_train, nout)
    test_set = ArrayIterator(X_test, y_test, nout)

    # Build the network
    (encoder, decoder), cost, opt = get_core_net(nout=nout, vocab_size=vocab_size,
                                                 dropout_rate = dropout_rate, embedding_size = embedding_size, activation_function = activation_function)

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
    file_prefix = './saved_models/' + args.data_source
    save_core(file_prefix,encoder, decoder)

