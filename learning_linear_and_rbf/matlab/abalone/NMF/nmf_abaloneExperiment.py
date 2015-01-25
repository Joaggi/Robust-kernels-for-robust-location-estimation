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
import scipy.io as sio

#parameters
fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/abalone.npz'
epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/parameters.mat')['epocs']
n_clusters = 3

clustering_accuracy = np.zeros(epocs)
calculate_purity = np.zeros(epocs)
calculate_nmi = np.zeros(epocs)

for epocs in xrange(epocs):
   labels_true,labels_pred,features= NMF.get_NMF(fileData,n_clusters, normalized_axis=0,norm = 'l1')


   clustering_accuracy[epocs] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
   calculate_purity[epocs],vector = Metrics.calculate_purity(labels_true,labels_pred)
   calculate_nmi[epocs] = Metrics.calculate_nmi(labels_true,labels_pred)
   
sio.savemat('results',{'clusteringAccuracy' : clustering_accuracy,'purityvec' : calculate_purity,'nmivec' : calculate_nmi})

