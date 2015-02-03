# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:41:46 2014

@author: Alejandro
"""
import os
os.chdir("G:\Dropbox\Universidad\Machine Learning\Robustes\Corel2500")

from sklearn.cluster import k_means
import numpy as np
from KernelKMeans import KernelKMeans
from CorelExperiment import KernelMetrics


'''
Initializations
'''
kernelMetrics = KernelMetrics(  n_clusters=20, n_clusters_outliers = 5 , n_experiment = 1)
kernelMetrics.init_figure()   
kernelMetrics.defineVariablesCorel()

    
X , y, X_contamination , n_samples, n_outliers,  n_features , n_clusters = \
                                kernelMetrics.load_data()

epochs = n_outliers

mean_KMeans_initial , mean_KernelKMeans_initial , KMeans_labels, KernelKMeans_labels = \
                                kernelMetrics.initial_means(X,gamma =2**-3)
X_with_contamination = np.array([[X],[X_contamination]])


kernelMetrics.calculate_purity(KMeans_labels,0)
kernelMetrics.calculate_purity(KernelKMeans_labels,0,0)
kernelMetrics.calculate_entropy(KMeans_labels,0)
kernelMetrics.calculate_entropy(KernelKMeans_labels,0,0)


for epoch in range(1,2):
    print epoch
    for n_iter in range(kernelMetrics.n_iter):        
        X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,epoch)
        
        mean_KMeans,KMeans_labels, intertia = k_means(X_with_contamination, n_clusters, random_state = 1)
        #print "Media Kmeans: " + str(mean_KMeans)
        kernelMetrics.calculate_purity(KMeans_labels,epoch,n_iter)
        kernelMetrics.calculate_entropy(KMeans_labels,epoch,n_iter)
        
        for i in range(1,5):
            KernelKMeans_model = KernelKMeans(n_clusters=n_clusters, random_state=1,
                             kernel="rbf", gamma=2**-i, coef0=1,
                             verbose=0)
                             
            KernelKMeans_model.fit(X_with_contamination)
            mean_KernelKMeans = kernelMetrics.pre_image(X_with_contamination, KernelKMeans_model.labels_, 
                             KernelKMeans_model.gamma, 100)
            
            
            kernelMetrics.calculate_bias(
                                mean_KMeans,mean_KMeans_initial,mean_KernelKMeans,mean_KernelKMeans_initial,epoch,n_iter)
                                                            
            kernelMetrics.calculate_entropy(KernelKMeans_model.labels_,epoch,i,n_iter)
            
            kernelMetrics.calculate_purity(KernelKMeans_model.labels_,epoch,i,n_iter)


for i in range(1,5):
    kernelMetrics.create_graphs_bias(i, 5)
    
for i in range(1,5):
    kernelMetrics.create_graphs_purity(np.sum(kernelMetrics.purity['kmeans'],axis=1)/kernelMetrics.n_clusters,np.sum(kernelMetrics.purity['kernelkmeans'][:,i,:],axis=1)/n_clusters,i, 5)
    
for i in range(1,5):
    kernelMetrics.create_graphs_entropy(kernelMetrics.calculate_entropy['kmeans'],kernelMetrics.calculate_entropy['KernelKMeans'][:,i],i, 5)