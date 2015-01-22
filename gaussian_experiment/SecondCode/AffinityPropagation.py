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
from sklearn.preprocessing import normalize




X = normalize(X, norm='l2', axis= 0)

model_affinitypropagation = AffinityPropagation()
model_affinitypropagation.fit(X)
len(model_affinitypropagation.cluster_centers_indices_)
kernel = pairwise_kernels(X,metric="rbf")
model_affinitypropagation = AffinityPropagation(affinity='precomputed')
model_affinitypropagation.fit(kernel)
len(model_affinitypropagation.cluster_centers_indices_)

kernelMetrics = KernelMetrics(  n_clusters=20, n_clusters_outliers = 5 , n_experiment = 1)

X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,100,100)

X_with_contamination = normalize(X_with_contamination, norm='l2', axis= 0)

model_affinitypropagation = AffinityPropagation()
model_affinitypropagation.fit(X_with_contamination)
len(model_affinitypropagation.cluster_centers_indices_)
print model_affinitypropagation.cluster_centers_indices_

kernel = pairwise_kernels(X_with_contamination,metric="rbf")
model_affinitypropagation = AffinityPropagation(affinity='precomputed')
model_affinitypropagation.fit(kernel)
len(model_affinitypropagation.cluster_centers_indices_)
print model_affinitypropagation.cluster_centers_indices_

