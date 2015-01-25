# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:21:36 2014

@author: 
"""

import numpy as np
from numba import jit

@jit('pyobject(double[:,:],int32)')
def normalizeByRange(X, axis = 1):
    """
    Compute the normalize by range of a matrix by feature or by sample
    
    X = (X - X_min) / (X_max-X_min)

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    axis = 1 or zero
    
    Returns
    -------
    normalize_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X = np.array(X)
    X = X.astype(np.float)
    if axis == 1:
        X = X - np.min(X,axis=axis)[:,np.newaxis]
        rango = (np.max(X,axis=axis)-np.min(X,axis=axis))[:,np.newaxis]
    if axis == 0:
        X = X - np.min(X,axis=axis)[np.newaxis]  
        rango = (np.max(X,axis=axis)-np.min(X,axis=axis))[np.newaxis]
    return X/rango
    
def test():
    a = [[5.,4.,4.],[6.,1.,5.]]
    print normalizeByRange(a,1)
    
#test()