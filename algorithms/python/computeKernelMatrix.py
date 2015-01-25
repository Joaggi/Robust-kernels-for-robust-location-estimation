# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 20:23:24 2014

@author: Alejandro
"""
from gaussianKernel import gaussianKernel
#from kernel import kernelAndrews
#from kernel import kernelTukey

def computeKernelMatrix(X,Y,option):
    """
    Compute the different kernels of a matrix X and Y
    
    Parameters
    ----------
    X : array of shape (n_features , n_samples_X)
    Y : array of shape (n_features , n_samples_X)
    option = dictionary
    
    Returns
    -------
    kernel : array of shape (n_samples_X, n_samples_Y)
    """
    if option['kernel']  == 'rbf':
        if option['param'] == None:
            option['param'] = 2
        return gaussianKernel(X,Y,option['param'])  
        
#    if option['kernel']  == 'andrews':
#        if option['param'] == None:
#            option['param'] = 1
#        return kernelAndrews.compute(X,X,option)
            
#    if option['kernel']  == 'andrews':
#        if option['param'] == None:
#            option['param'] = 1
        
        
    