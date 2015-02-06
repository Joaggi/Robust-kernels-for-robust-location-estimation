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
        c = X.mean()*50
    print c
    K = euclidean_distances(X, Y, squared=False)
    
    print K
    print "Shape kernel" + str(np.where(K/c > 1)[0].shape)
    print (K/c)
    print "power"
    print np.pi*(K/c)
    print "power 2"
    print np.cos(np.pi*(K/c))
    print "power 3"
    print 1.0/2.0*np.power(np.pi,2) *np.cos(np.pi*(K/c)) 
    gramMatrix = np.zeros(K.shape)
    gramMatrix[np.where(K/c <= 1)] = (1.0/2.0*np.power(np.pi,2) *np.cos(np.pi*(K/c))) [np.where(K/c <= 1)]
    gramMatrix[np.where(K/c > 1)] = -1.0/2.0*np.power(np.pi,2)
    print "Shape kernel" + str(np.where(K/c > 1)[0].shape)
    return gramMatrix


vectorProof = np.random.uniform(-99999999,99999999,[1000,7])
gramMatrix = andrews_kernel(vectorProof)
print gramMatrix
U, s, V = np.linalg.svd(gramMatrix, full_matrices=False)
print s
print np.all(s>0)
