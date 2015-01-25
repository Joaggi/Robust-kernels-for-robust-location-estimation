# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""

import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning/Algorithms/python')
sys.path.append(lib_path)
import KernelKMeans 
import numpy as np
import Metrics
import scipy.io as sio

#parameters
fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/abalone.npz'
epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/Abalone/parameters.mat')['epocs']
n_clusters = 3

gamma_logscale = [1,2,3,4]

clustering_accuracy = np.zeros(epocs*4)
calculate_purity = np.zeros(epocs*4)
calculate_nmi = np.zeros(epocs*4)

cont = 0
for gamma in gamma_logscale:
    for epocs in xrange(epocs):
        print epocs
        labels_true,labels_pred,features= KernelKMeans.get_kernelKMeans(fileData,n_clusters,normalized_axis = 0,norm='l1',gamma=2**-gamma)
        #print str(np.where(labels_true==0)[0].shape) + ' ' + str(np.where(labels_pred==0)[0].shape)
        #print str(np.where(labels_true==1)[0].shape) + ' ' + str(np.where(labels_pred==1)[0].shape)
        #print str(np.where(labels_true==2)[0].shape) + ' ' + str(np.where(labels_pred==2)[0].shape)
    
        clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
        calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
        calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
        cont +=1

sio.savemat('results',{'clusteringAccuracy' : clustering_accuracy,'purityvec' : calculate_purity,'nmivec' : calculate_nmi})