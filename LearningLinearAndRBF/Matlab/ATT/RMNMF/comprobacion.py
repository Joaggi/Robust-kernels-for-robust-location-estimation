# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:20:40 2014

@author: Alejandro
"""
import os, sys
lib_path = os.path.abspath('../../../Algorithms/python')
sys.path.append(lib_path)
import numpy as np
import Metrics
import scipy.io as sio

data = sio.loadmat('labels.mat')
labels_true= data['labels'].reshape((2600))
labels_pred = data['labels_pred'].reshape((2600))

clustering_accuracy[epocs] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
calculate_purity[epocs],vector = Metrics.calculate_purity(labels_true,labels_pred)
calculate_nmi[epocs] = Metrics.calculate_nmi(labels_true,labels_pred)