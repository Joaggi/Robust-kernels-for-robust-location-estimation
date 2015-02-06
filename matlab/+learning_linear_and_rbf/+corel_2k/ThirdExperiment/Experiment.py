# -*- coding: utf-8 -*-
"""
Created on Sun Jun 08 19:02:24 2014

@author: Alejandro
"""

import sys
sys.path.append("G:\Dropbox\Universidad\Machine Learning\Algorithms\Python")

from CorelFunctions import load_data
from KernelKMeans import KernelKMeans
from Metrics import calculate_purity
from sklearn.cluster import KMeans

X ,y,X_contamination , n_samples, n_outliers, n_features , n_clusters = \
        load_data(10,0,norm='l1',byAxis = 0)

KernelKMeans_model = KernelKMeans(n_clusters=n_clusters,
                     kernel="rbf", gamma=2**-3, coef0=1,
                     verbose=0,max_iter=10000)        
                     
KernelKMeans_model.fit(X)

KMeans_model = KMeans(n_clusters)
KMeans_model.fit(X)

purityKernelKMeans,purityVectorKernelKMeans = calculate_purity(KernelKMeans_model.labels_,y,n_clusters)
purityKMeans,purityVectorKernelKMeans = calculate_purity(KMeans_model.labels_,y,n_clusters)