# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:59:45 2014

@author: Alejandro
"""

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import check_pairwise_arrays

import numpy as np

def andrews_kernel(X, Y=None, c=None):
    """
    Compute the andrews kernel between X and Y::

        K(x, y) = { -c/2 * (1 - cos(||x-y||/c)),  if ||x-y||/c <= pi
                    -c,                            if ||x-y||/c >  pi
                                            }

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """

    X, Y = check_pairwise_arrays(X, Y)

    if c is None:
        c = X.mean()*50
    K = euclidean_distances(X, Y, squared=False)

    gramMatrix = np.zeros(K.shape)
    gramMatrix[np.where(K/c <= np.pi)] = (-c/2.0 * (1 - np.cos(K/c))) [np.where(K/c <= np.pi)]
    gramMatrix[np.where(K/c > np.pi)] = -c
    return gramMatrix
