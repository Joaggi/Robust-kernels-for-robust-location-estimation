# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:51:40 2014

@author: Alejandro
"""
import os
os.chdir("G:\Dropbox\Universidad\Machine Learning\Robustes\Corel2500")

from sklearn.metrics.pairwise import pairwise_kernels
from CorelExperiment import KernelMetrics
from sklearn.cluster import AffinityPropagation

import numpy as np
from KernelKMeans import KernelKMeans
from CorelExperiment import KernelMetrics

'''
Initializations
'''
kernelMetrics = KernelMetrics(  n_clusters=20, n_clusters_outliers = 5 , n_experiment = 1)
    
X , y, X_contamination , n_samples, n_outliers,  n_features , n_clusters = \
                                kernelMetrics.load_data()


model_affinitypropagation = AffinityPropagation(preference = np.percentile(X,q = 0.0001)-0.135)
model_affinitypropagation.fit(X)
len(model_affinitypropagation.cluster_centers_indices_)
print "SED: " + str(model_affinitypropagation.cluster_centers_indices_)

kernel = pairwise_kernels(X,metric="rbf")
model_affinitypropagation = AffinityPropagation(affinity='precomputed',preference = np.percentile(kernel,q = 0.318))
model_affinitypropagation.fit(kernel)
len(model_affinitypropagation.cluster_centers_indices_)
print "KERNEL: " + str(model_affinitypropagation.cluster_centers_indices_)

kernelMetrics = KernelMetrics(  n_clusters=20, n_clusters_outliers = 5 , n_experiment = 1)

X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,30)

model_affinitypropagation = AffinityPropagation(preference = np.percentile(X_with_contamination,q = 0.0001)-0.135)
model_affinitypropagation.fit(X_with_contamination)
len(model_affinitypropagation.cluster_centers_indices_)
print "SED: " + str(model_affinitypropagation.cluster_centers_indices_)
kernel = pairwise_kernels(X_with_contamination,metric="rbf")
model_affinitypropagation = AffinityPropagation(affinity='precomputed',preference = np.percentile(kernel,q = 0.318))
model_affinitypropagation.fit(kernel)
len(model_affinitypropagation.cluster_centers_indices_)
print "KERNEL: " + str(model_affinitypropagation.cluster_centers_indices_)

