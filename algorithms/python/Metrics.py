# -*- coding: utf-8 -*-
"""
Created on Sun Jun 08 18:27:19 2014

@author: Alejandro
"""
import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning/')
sys.path.append(lib_path)
import numpy as np
from scipy.stats import mode
from sklearn.utils import as_float_array
from sklearn.metrics import normalized_mutual_info_score
from Algorithms.Python.munkres.munkres import Munkres, print_matrix
from sklearn.metrics.cluster import contingency_matrix
import sys

def calculate_purity(labels_true,labels_pred):
    labels_true = np.array(labels_true)
    labels_true = labels_true.reshape(labels_true.size)
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.reshape(labels_pred.size)
    
    k = np.size(np.unique(labels_pred))
    purityVector = np.zeros(k)
    purity = 0
    matrix = as_float_array(contingency_matrix(labels_true,labels_pred))
    for i in xrange(k):
        moda = np.float(np.max(matrix[:,i]))
        purityVector[i] = moda / np.sum(matrix[:,i])
        purity += purityVector[i] * np.sum(matrix[:,i]) / np.size(labels_pred)
    return purity,purityVector
        
def calculate_nmi(labels_true,labels_pred):
    labels_true = np.array(labels_true)
    labels_true = labels_true.reshape(labels_true.size)
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.reshape(labels_pred.size)
    
    labels_true = as_float_array(labels_true)
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.reshape(labels_pred.size)
    return normalized_mutual_info_score(labels_true, labels_pred)
    
def calculate_clusteringAccuracy(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_true = labels_true.reshape(labels_true.size)
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.reshape(labels_pred.size)
    
    matrix = contingency_matrix(labels_true,labels_pred)
    return get_IndicesClusterxClass(matrix)

def get_IndicesClusterxClass(matrix):
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]
    
    m = Munkres()
    indexes = m.compute(cost_matrix)
    #print indexes
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
    return float(total) / float(matrix.sum())



        
