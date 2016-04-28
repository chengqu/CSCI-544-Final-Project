#!/usr/bin/env python

"""
File that will load the trained weights of the core network, and train a second
encoder from a different datasource.
"""


from neon.backends import gen_backend
from neon.backends.backend import Block
from neon.data import ArrayIterator
from neon.models import Model
from neon.layers.container import Sequential, MergeMultistream, MergeBroadcast
from neon.callbacks.callbacks import Callbacks
from neon.transforms import Accuracy
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.layers import RecurrentSum, RecurrentMean

from networks import load_core, get_title_augmentor
from data import get_imdb, get_saudinet_data


class MultiModalModel(Model):

    def _epoch_fit(self, dataset, callbacks):
        epoch = self.epoch_index
        self.total_cost[:] = 0
        for mb_idx, (x, t) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)
            x = self.fprop(x)
            self.total_cost[:] = self.total_cost + self.cost.get_cost(x, t)

            # deltas back propagate through layers
            # for every layer in reverse except the 0th one
            delta = self.cost.get_errors(x, t)

            self.bprop(delta)
            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)
            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)
        self.total_cost[:] = self.total_cost / dataset.nbatches


if __name__ == '__main__':
    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    parser.add_argument("new_data_source", help = "choose the data source used for classification. options: content, title, author, author-content, author-title, content-title")
    parser.add_argument("initial_model", help = "choose the initial encoder-decoder model. options: content, title, author")
    

    args = parser.parse_args(gen_be=False)

    # hyperparameters from the reference
    args.batch_size = 32
    args.backend = 'gpu'
    
    print ' '
    print "new source used for classification is: ", args.new_data_source
    print "initial model is trained on: ", args.initial_model
    print ' '
    
    
    # Setup backend
    be = gen_backend(**extract_valid_args(args, gen_backend))

    # Load the dataset
    (X_train_content, y_train_content), (X_test_content, y_test_content), nout, vocab_size = get_saudinet_data(args, modality='content')
    (X_train_title, y_train_title), (X_test_title, y_test_title), nout, vocab_size = get_saudinet_data(args, modality='title')
    (X_train_authors, y_train_authors), (X_test_authors, y_test_authors), nout, vocab_size = get_saudinet_data(args, modality='author')
    
    
    X_train, X_test = {}, {}
    X_train['content-author'], X_test['content-author'] = [X_train_content,X_train_authors], [X_test_content,X_test_authors]
    X_train['content-title'], X_test['content-title'] = [X_train_content,X_train_title], [X_test_content,X_test_title]
    
    X_train['title-author'], X_test['title-author'] = [X_train_title,X_train_authors], [X_test_title, X_test_authors]
    X_train['title-content'], X_test['title-content'] = [X_train_title,X_train_content], [X_test_title, X_test_content]
    
    X_train['author-content'], X_test['author-content'] = [X_train_authors, X_train_content], [X_test_authors, X_test_content]
    X_train['author-title'], X_test['author-title'] = [X_train_authors, X_train_title], [X_test_authors, X_test_title]

    X_train['content-author-title'], X_test['content-author-title'] = [X_train_content,X_train_authors,X_train_title], [X_test_content,X_test_authors,X_test_title]

    X_train['title-author-content'], X_test['title-author-content'] = [X_train_title,X_train_authors,X_train_content], [X_test_title, X_test_authors,X_test_content]

    X_train['author-content-title'], X_test['author-content-title'] = [X_train_authors, X_train_content,X_train_title], [X_test_authors, X_test_content,X_test_title]


    
    
    data_source_keyword = args.initial_model + '-' + args.new_data_source
    
    
    train_set = ArrayIterator( X_train[data_source_keyword], y_train_content, nout)
    test_set = ArrayIterator(  X_test[data_source_keyword],  y_test_content,  nout)
    
    
    
    #train_set = ArrayIterator([X_train_content, X_train_title], y_train_content, nout)
    #test_set = ArrayIterator([X_test_content, X_test_title], y_test_content, nout)

    # Build the network
    file_prefix = './saved_models/' + args.initial_model
    encoder, decoder = load_core(file_prefix)
    encoder = Sequential(encoder)
    augmentor, cost, opt = get_title_augmentor(vocab_size=vocab_size)
    augmentor = Sequential(augmentor)
    model = MultiModalModel(layers=[
        MergeMultistream(layers=[encoder, augmentor], merge='recurrent'),
        RecurrentMean(),
        decoder
    ])

    callbacks = Callbacks(model, **args.callback_args)

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

