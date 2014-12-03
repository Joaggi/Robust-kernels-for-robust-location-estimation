# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:36:24 2014

@author: Alejandro
"""

from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import as_float_array

from numba import jit

@jit(nopython=True)
def get_kMeans(fileData,normalized_axis=1,norm = 'l1'):
    data = np.load(fileData)
    #print data
    features = data['data']
    n_clusters = np.size(np.unique(data['labels']))
    features = as_float_array(features)
    #print features
    if normalized_axis != None:
        features = normalize(features,norm=norm,axis=normalized_axis)
    print features
    model = KMeans(n_clusters = n_clusters,tol=1e-2)
    print model
    model.fit(features)
    print model.labels_
    labels_pred = model.labels_
    labels_true = data['labels']
    return labels_true, labels_pred,features
