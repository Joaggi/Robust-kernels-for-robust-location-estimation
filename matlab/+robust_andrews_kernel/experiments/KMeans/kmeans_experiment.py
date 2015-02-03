# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""
import os, sys
lib_path = os.path.abspath('G:\Dropbox\Universidad\Machine Learning')
sys.path.append(lib_path)
import Algorithms.Python.KMeans as KMeans
import numpy as np
import Algorithms.Python.Metrics as Metrics
import scipy.io as sio
from numba import jit

@jit()
def kmeans(fileData, epocs ,normalized_axis = 0 ,norm='l1'):
    
	#parameters
	epocs = sio.loadmat(epocs)['epocs'][0][0]

	clustering_accuracy = np.zeros(epocs)
	calculate_purity = np.zeros(epocs)
	calculate_nmi = np.zeros(epocs)

	for epocs in xrange(epocs):
         print epocs
         labels_true,labels_pred,features= KMeans.get_kMeans(fileData,normalized_axis = normalized_axis ,norm=norm)


         clustering_accuracy[epocs] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
         calculate_purity[epocs],vector = Metrics.calculate_purity(labels_true,labels_pred)
         calculate_nmi[epocs] = Metrics.calculate_nmi(labels_true,labels_pred)
	   
	sio.savemat('results',{'clusteringAccuracy' : clustering_accuracy,'purityvec' : calculate_purity,'nmivec' : calculate_nmi})
