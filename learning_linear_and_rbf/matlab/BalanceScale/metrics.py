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
import Robustes.Experiments.metrics_over_labels as metrics_over_labels
    
dictionary = {
    'Kernelconvexnmf':0,
    'KernelKMeans':0,
    'Kernelseminmfnnls':0,
    'Kernelseminmfrule':0,
    'KMeans':0,
    'NNMF':0,
    'RMNMF':1
    }
labels_name = 'balance-scale-labels'
metrics_over_labels.metrics('G:/Dropbox/Universidad/Machine Learning/Robustes/BalanceScale/',dictionary,labels_name)