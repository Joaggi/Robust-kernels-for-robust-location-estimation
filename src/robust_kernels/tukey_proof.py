# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:59:45 2014

@author: Alejandro
"""

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import check_pairwise_arrays

import numpy as np

def tukey_kernel(X, Y=None, c=None):
    """
    Compute the tukey kernel between X and Y::

        K(x, y) = {1/12*(1-(||x-y||/c)**2)**3
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
        c = X.mean()*5
    K = euclidean_distances(X, Y, squared=False)
    
    gramMatrix = np.zeros(K.shape)
    gramMatrix[np.where(K/c <= 1)] = ((1.0/2.0) * np.power(1 - np.power((K/c),2),3))[np.where(K/c <= 1)]
    gramMatrix[np.where(K/c > 1)] = 0
    return gramMatrix

