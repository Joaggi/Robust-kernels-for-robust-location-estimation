# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 20:55:38 2014

@author: Alejandro
"""

from sklearn.decomposition import NMF
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import as_float_array

def get_NMF(fileData,normalized_axis=1,norm='l2'):
    data = np.load(fileData)

    n_components = np.size(np.unique(data['labels']))
    features = data['data']
    print features
    features = as_float_array(features)
    if normalized_axis != None:
        features = normalize(features,norm=norm,axis=normalized_axis)
        
    model = NMF(n_components = n_components)
    print n_components
    print model
    model.fit(np.transpose(features))
    G = model.components_
    print G
    labels_pred = np.zeros(features.shape[0])
    for i in xrange(G.shape[1]):
        temp2= np.argmax(G[:,i])
        labels_pred[i] = temp2
    labels_true = data['labels']
    return labels_true,labels_pred,features
    
