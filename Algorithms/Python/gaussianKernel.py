# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:21:36 2014

@author: 
"""

import numpy as np
from numba import jit

def gaussianKernel(X,Y=None,sigma=1):
    if Y == None:
        Y = X
    if sigma == 0:
        raise Exception("Sigma can't be zero")
    X = X.astype(float)    
    Y = Y.astype(float)
    return getKernel(X,Y,sigma)
    
#@jit('pyobject(double[:,:],double[:,:],float32)')
def getKernel(X,Y,sigma):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-sigma ||x-y||**2)

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array of shape (n_features, n_samples)

    Y : array of shape (n_features, n_samples)

    gamma : float

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """

    dot1 = np.dot(np.diag(np.dot(np.transpose(X),X))[:,np.newaxis],np.ones([1,Y.shape[1]]))
    dot2 = np.dot(np.ones([X.shape[1],1]),np.transpose(np.diag(np.dot(np.transpose(Y),Y))[:,np.newaxis]))
    K = dot1 + dot2 - 2.*np.dot(np.transpose(X),Y)
    K = np.exp((-1./sigma**2)*K)
    return K
    
def test():
    fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/AR/AR.npz'
    epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/AR/parameters.mat')['epocs']
    data = np.load(fileData)
    X = data['data']
    print gaussianKernel(X,X,2)
    