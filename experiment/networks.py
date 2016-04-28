#!/usr/bin/env python

"""
Defines the network architecture, hyper-params, optimizer and cost to be used
in the experiements.
"""

from neon.models import Model
from neon.initializers import GlorotUniform, Uniform
from neon.optimizers import Adagrad
from neon.layers import (
    LSTM,
    Linear,
    Affine,
    LookupTable,
    RecurrentSum,
    GeneralizedCost,
    Dropout,
)
from neon.transforms import (
    Tanh,
    Logistic,
    Softmax,
    CrossEntropyMulti,
    Explin,
    Normalizer,
)




def get_cost_opt():
    gradient_clip_value = 15.0
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
    opt = Adagrad(learning_rate=0.01, gradient_clip_value=gradient_clip_value)
    return cost, opt


def get_core_net(embedding_size=128, vocab_size=20000, nout=10, dropout_rate = 0.5, activation_function = 1):
    """
    Returns a tuple containing the original encoder and decoder networks.
    embedd_size: size of the output of the encoder
    nout: size of the output of the decoder
    path: filepath from which the weights should be loaded
    """
    if activation_function == 1:
        af = Tanh()
    if activation_function == 2:
        af = Explin(alpha=0)
    if activation_function == 3:
        af = Logistic()
    if activation_function == 4:
        af = Normalizer()


    embedding_dim = embedding_size
    uni = Uniform(low=-0.1/embedding_dim, high=0.1/embedding_dim)
    glorot = GlorotUniform()

    enc = [
        LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=uni, pad_idx=0, update=True),
        LSTM(embedding_size, glorot, activation=af, gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
    ]

    dec = [
        Dropout(keep=0.25),
        Dropout(keep=dropout_rate),
        Affine(nout, glorot, bias=glorot, activation=Softmax())
    ]

    cost, opt = get_cost_opt()
    return (enc, dec), cost, opt


def load_core(file_prefix, embedding_size=128):
    enc = Model(file_prefix + '_encoder.neon').layers.layers
    dec = Model(file_prefix + '_decoder.neon').layers.layers
    return enc, dec


def save_core(file_prefix,encoder, decoder):
    Model(encoder).save_params(file_prefix + '_encoder.neon', True)
    Model(decoder).save_params(file_prefix + '_decoder.neon', True)


def get_title_augmentor(vocab_size=20000, embedding_size=128, path=None):
    """
    Returns the network that will be used to augment the network, to handle
    titles.
    embedd_size: size of the embedding
    path: filepath from which to load the model's weights
    """
    uni = Uniform(low=1.0, high=1.0)
    glorot = GlorotUniform()

    aug = [
        LookupTable(vocab_size=vocab_size, embedding_dim=128, init=uni),
        LSTM(embedding_size, glorot, activation=Tanh(),
             gate_activation=Logistic(), reset_cells=True),
        RecurrentSum(),
    ]
    cost, opt = get_cost_opt()
    return aug, cost, opt
