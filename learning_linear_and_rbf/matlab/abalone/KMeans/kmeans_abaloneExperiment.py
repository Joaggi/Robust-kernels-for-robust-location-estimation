# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""

import os, sys
#lib_path = os.path.abspath('../../../Algorithms/python')

lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')

sys.path.append(lib_path)
import Algorithms.Python.KMeans as KMeans
import numpy as np
import Algorithms.Python.Metrics as Metrics
import scipy.io as sio

#parameters
#fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/abalone.npz'
#epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/parameters.mat')['epocs']

fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/abalone.npz'
epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/parameters.mat')['epocs']


n_clusters = 3

clustering_accuracy = np.zeros(epocs)
calculate_purity = np.zeros(epocs)
calculate_nmi = np.zeros(epocs)
labels_total = np.zeros((epocs,4177))

for epocs in xrange(epocs):
   labels_true,labels_pred,features= KMeans.get_kMeans(fileData,normalized_axis = 0 ,norm='l1')
   labels_total[epocs,:] = labels_pred

   clustering_accuracy[epocs] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
   calculate_purity[epocs],vector = Metrics.calculate_purity(labels_true,labels_pred)
   calculate_nmi[epocs] = Metrics.calculate_nmi(labels_true,labels_pred)

clustering_accuracy_mean = np.mean(clustering_accuracy)
calculate_purity_mean =  np.mean(calculate_purity)
calculate_nmi_mean =  np.mean(calculate_nmi)

sio.savemat('results',{'labels_total' : labels_total ,'clusteringAccuracyVec' : clustering_accuracy,'purityVec' : calculate_purity,'nmiVec' : calculate_nmi , 'clusteringAccuracyMeanVec' : clustering_accuracy_mean, 'purityMeanVec' : calculate_purity_mean, 'nmiMeanVec' : calculate_nmi_mean})
