# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:50:05 2014

@author: Alejandro
"""

import scipy.io as sio
import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)

import numpy as np
import Algorithms.Python.Metrics as Metrics

labels_pred_total = sio.loadmat('MovementLibras/Kernelseminmfnnls/results')['labels_total']
labels_true = sio.loadmat('MovementLibras/movement_libras_labels')['labels']

print labels_pred_total.shape
print labels_true.shape

clustering_accuracy = np.zeros(21*30)
calculate_purity = np.zeros(21*30)
calculate_nmi = np.zeros(21*30)

clustering_accuracy_mean = np.zeros(21)
calculate_purity_mean = np.zeros(21)
calculate_nmi_mean = np.zeros(21)

cont = 0
for j in range(21):
    aux = cont
    for i in range(30):
        labels_pred = labels_pred_total[cont,:]
        clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
        calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
        calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
        cont+=1
    clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
    calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
    calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
    

sio.savemat('MovementLibras/Kernelseminmfnnls/results2',{'clusteringAccuracyVec' : clustering_accuracy,'purityVec' : calculate_purity,'nmiVec' : calculate_nmi , 'clusteringAccuracyMeanVec' : clustering_accuracy_mean, 'purityMeanVec' : calculate_purity_mean, 'nmiMeanVec' : calculate_nmi_mean})