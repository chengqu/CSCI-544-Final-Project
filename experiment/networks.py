#!/usr/bin/env python

"""
Defines the network architecture, hyper-params, optimizer and cost to be used
in the experiements.
"""


def get_core_net(embedd_size=1024, nout=10, path=None):
    """
    Returns a tuple containing the original encoder and decoder networks.
    embedd_size: size of the output of the encoder
    nout: size of the output of the decoder
    path: filepath from which the weights should be loaded
    """
    cost = None
    opt = None
    model = None
    if path is not None:
        model.load_params(path)
    return model, cost, opt


def get_title_augmentator(embedd_size=1024, path=None):
    """
    Returns the network that will be used to augment the network, to handle
    titles.
    embedd_size: size of the embedding
    path: filepath from which to load the model's weights
    """
    cost = None
    opt = None
    model = None
    if path is not None:
        model.load_params(path)
    return model, cost, opt
