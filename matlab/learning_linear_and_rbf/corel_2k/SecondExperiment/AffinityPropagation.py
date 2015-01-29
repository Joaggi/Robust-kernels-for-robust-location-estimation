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

'''
Initializations
'''

def defineVariablesCorel():
    bias = {}
    purity= {}
    entropy = {}
    calculate_entropy = {}
    model = {}   
    bias['kernelkmeans']=np.zeros((n_outliers+1,11))
    bias['kmeans'] = np.zeros(n_outliers+1)
    purity['kmeans'] = np.zeros((n_outliers+1,n_clusters))
    purity['kernelkmeans'] = np.zeros((n_outliers+1,n_clusters,5))
    entropy['kmeans'] = np.zeros((n_outliers+1,n_clusters))
    calculate_entropy['kmeans'] = np.zeros(n_outliers+1)
    entropy['KernelKMeans'] = np.zeros((n_outliers+1,n_clusters,5))
    calculate_entropy['KernelKMeans'] = np.zeros((n_outliers+1,5))    
    return bias , purity ,entropy, calculate_entropy, model

def train_model( X,  quantile, shift = 0, isKernel = False):
    if isKernel == False:
        preference = np.percentile(X,q = quantile)-shift
        model_affinityPropagation = AffinityPropagation(preference = preference)
        model_affinityPropagation.fit(X)
        return model_affinityPropagation
    else:
        kernel = pairwise_kernels(X,metric="rbf")
        preference = np.percentile(X,q = quantile)-shift
        model_affinityPropagation = AffinityPropagation(affinity='precomputed',preference = np.percentile(kernel,q = 0.318))
        model_affinityPropagation.fit(kernel)
        return model_affinityPropagation        
    
    
kernelMetrics = KernelMetrics(  n_clusters=20, n_clusters_outliers = 5 , n_experiment = 1)
    
X , y, X_contamination , n_samples, n_outliers,  n_features , n_clusters = \
                                kernelMetrics.load_data()

bias , purity, entropy, calculate_entropy , model = defineVariablesCorel()

model['sed_without_contamination'] = train_model(X,quantile = 0.0001 , shift = 0.135)
len(model['sed_without_contamination'].cluster_centers_indices_)
print "SED: " + str(model['sed_without_contamination'].cluster_centers_indices_)

model['kernel_without_contamination'] = train_model(X,isKernel = True , quantile = 0.318 )
len(model['kernel_without_contamination'].cluster_centers_indices_)
print "KERNEL: " + str(model['kernel_without_contamination'].cluster_centers_indices_)

X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,30)

model['sed_with_contamination'] = train_model(X_with_contamination ,quantile = 0.0001 , shift = 0.135)
len(model['sed_with_contamination'].cluster_centers_indices_)
print "SED: " + str(model['sed_with_contamination'].cluster_centers_indices_)

model['kernel_with_contamination']= train_model(X_with_contamination,isKernel = True , quantile = 0.318 )
len(model['kernel_with_contamination'].cluster_centers_indices_)
print "KERNEL: " + str(model['kernel_with_contamination'].cluster_centers_indices_)

