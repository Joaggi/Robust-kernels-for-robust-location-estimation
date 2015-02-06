# -*- coding: utf-8 -*-
"""
Created on Sun Jun 08 18:55:53 2014

@author: Alejandro
"""
import sys
sys.path.append("G:\Dropbox\Universidad\Machine Learning\Algorithms\Python")

import numpy as np
from scipy import io
from sklearn.preprocessing import normalize

def load_data(n_clusters, n_clusters_outliers, norm='l1',byAxis = 0):
    n_datos = n_clusters * 100
    n_outliers = n_clusters_outliers * 100
    
    # Load the data
    data_total = io.loadmat('Data/dat.mat')
    # Assign the data to the variable
    data_full = data_total['dat'][:,0:(n_datos+n_outliers)] 
    
    data_full = normalize(data_full, norm=norm, axis= byAxis)
    
    X = np.transpose(data_full[:,0:n_datos])
    X_contamination = np.transpose(data_full[:,n_datos:n_datos+n_outliers])
    
    y = np.zeros(n_datos)
    aux = 0
    for i in range(n_datos):
        y[i] = aux
        if i %100 == 0:
            aux+=1
            
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    return X , y , X_contamination , n_samples, n_outliers, n_features , n_clusters