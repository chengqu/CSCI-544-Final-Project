#!/usr/bin/env python

"""
File responsible for downloading the data and setting up the single and
multi-data stream.
"""


def download_data():
    """
    Should automatically download the data from:
    https://github.com/ParallelMazen/SaudiNewsNet
    """
    pass


def get_data(size=None, datas=None):
    """
    Should return a tuple of ArrayIterators containing size samples, from the
    sources as specified in datas.
    """
    if len(datas) < 1:
        datas = ('content', )
    pass

