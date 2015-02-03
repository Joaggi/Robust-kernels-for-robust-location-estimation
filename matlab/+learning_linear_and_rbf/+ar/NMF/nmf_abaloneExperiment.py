# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:23:43 2014

@author: Alejandro
"""
import os, sys
lib_path = os.path.abspath('../../../Algorithms/python')
sys.path.append(lib_path)
import NMF 
import numpy as np
import Metrics

#parameters
fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/abalone.npz'
n_clusters = 3

labels_true,labels_pred,features= NMF.get_NMF(fileData,n_clusters, normalized_axis=0)
print str(np.where(labels_true==0)[0].shape) + ' ' + str(np.where(labels_pred==0)[0].shape)
print str(np.where(labels_true==1)[0].shape) + ' '+ str(np.where(labels_pred==1)[0].shape)
print str(np.where(labels_true==2)[0].shape) + ' ' + str(np.where(labels_pred==2)[0].shape)

clustering_accuracy = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
print clustering_accuracy
calculate_purity = Metrics.calculate_purity(labels_true,labels_pred)
print calculate_purity
calculate_nmi = Metrics.calculate_nmi(labels_true,labels_pred)
print calculate_nmi


