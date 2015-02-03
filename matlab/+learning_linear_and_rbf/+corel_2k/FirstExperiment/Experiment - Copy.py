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
    
X , y, X_contamination , n_samples, n_outliers,  n_features , n_clusters = \
                                kernelMetrics.load_data()
                                
bias_kernelkmeans=np.zeros((n_outliers+1,11))
bias_kmeans = np.zeros(n_outliers+1)
KMeans_purity = np.zeros((n_outliers+1,n_clusters))
KernelKMeans_purity = np.zeros((n_outliers+1,n_clusters,5))
KMeans_entropy = np.zeros((n_outliers+1,n_clusters))
KMeans_entropy_calculate = np.zeros(n_outliers+1)
KernelKMeans_entropy = np.zeros((n_outliers+1,n_clusters,5))
KernelKMeans_entropy_calculate = np.zeros((n_outliers+1,5))
 
epochs = n_outliers

mean_KMeans_initial , mean_KernelKMeans_initial , KMeans_labels, KernelKMeans_labels = \
                                kernelMetrics.initial_means(X)
X_with_contamination = np.array([[X],[X_contamination]])


KMeans_purity = kernelMetrics.calculate_purity(KMeans_purity,KMeans_labels,0)
KernelKMeans_purity = kernelMetrics.calculate_purity(KernelKMeans_purity,KernelKMeans_labels,0)
KMeans_entropy , KMeans_entropy_calculate =\
    kernelMetrics.calculate_entropy(KMeans_entropy,KMeans_entropy_calculate,KMeans_labels,0)
KernelKMeans_entropy,KernelKMeans_entropy_calculate = \
    kernelMetrics.calculate_entropy(KernelKMeans_entropy,KernelKMeans_entropy_calculate,KernelKMeans_labels,0,0)


for epoch in range(1,10):
    print epoch
    X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,epoch)
    
    mean_KMeans,KMeans_labels, intertia = k_means(X_with_contamination, n_clusters, random_state = 1)
    #print "Media Kmeans: " + str(mean_KMeans)
    KMeans_purity = kernelMetrics.calculate_purity(KMeans_purity,KMeans_labels,epoch)
    KMeans_entropy , KMeans_entropy_calculate =\
        kernelMetrics.calculate_entropy(KMeans_entropy,KMeans_entropy_calculate,KMeans_labels,epoch)
        
    #for i in range (n_clusters):
        #print "i:"  + str(np.where(KMeans_labels[0:1100] == i)[0].shape[0])
        
    
    for i in range(1,5):
        KernelKMeans_model = KernelKMeans(n_clusters=n_clusters, random_state=1,
                         kernel="rbf", gamma=2**-i, coef0=1,
                         verbose=0)
        KernelKMeans_model.fit(X_with_contamination)
        mean_KernelKMeans = kernelMetrics.pre_image(X_with_contamination, KernelKMeans_model.labels_, 
                         KernelKMeans_model.gamma, 100)
        
        
        bias_kmeans, bias_kernelkmeans[:,i] = kernelMetrics.calculate_bias(bias_kmeans,bias_kernelkmeans[:,i],
                                                        mean_KMeans,mean_KMeans_initial,
                                                        mean_KernelKMeans,mean_KernelKMeans_initial,
                                                        epoch)
                                                        
        KernelKMeans_entropy,KernelKMeans_entropy_calculate = kernelMetrics.calculate_entropy(KernelKMeans_entropy,KernelKMeans_entropy_calculate, KernelKMeans_model.labels_,epoch,i)
        
        KernelKMeans_purity = kernelMetrics.calculate_purity(KernelKMeans_purity,KernelKMeans_model.labels_,epoch,i)


kernelMetrics.init_figure()   


for i in range(1,5):
    kernelMetrics.create_graphs_bias(bias_kmeans,bias_kernelkmeans[:,i],i, 5)
    
for i in range(1,5):
    kernelMetrics.create_graphs_purity(np.sum(KMeans_purity,axis=1),np.sum(KernelKMeans_purity[:,i,:],axis=1),i, 5)/n_clusters
    
for i in range(1,5):
    kernelMetrics.create_graphs_entropy(KMeans_entropy_calculate,KernelKMeans_entropy_calculate[:,i],i, 5)