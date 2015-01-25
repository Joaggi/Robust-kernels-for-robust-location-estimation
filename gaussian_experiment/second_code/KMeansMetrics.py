# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:41:46 2014

@author: Alejandro
"""
import os
os.chdir("G:\Dropbox\Universidad\Machine Learning\Robustes\GaussianExperiment\SecondCode")

from sklearn.cluster import k_means
import numpy as np
from KernelKMeans import KernelKMeans

from GaussianExperiment import GaussianExperiment
from GaussianExperiment import KernelMetrics
    
kernelMetrics = KernelMetrics()
    
X , y, X_contamination , n_samples, n_outliers,  n_features , n_clusters = kernelMetrics.load_data(n_experiment = 1)
epochs = n_outliers
gaussianExperiment = GaussianExperiment()

bias_kernelkmeans=np.zeros((n_outliers+1,11))
bias_kmeans = np.zeros(n_outliers+1)

mean_KMeans_initial , mean_KernelKMeans_initial = kernelMetrics.initial_means(X,n_clusters)
X_with_contamination = np.array([[X],[X_contamination]])



for epoch in range(1,epochs):
    print epoch
    X_with_contamination , X_contamination_epoch = kernelMetrics.insert_contamination(X,X_contamination,epoch,n_outliers)
    
    mean_KMeans,label, intertia = k_means(X_with_contamination, n_clusters, random_state = 1)
    #print "Media Kmeans: " + str(mean_KMeans)
    
    for i in range(1,5):
        KernelKMeans_model = KernelKMeans(n_clusters=n_clusters, random_state=1,
                         kernel="rbf", gamma=2**-i, coef0=1,
                         verbose=0)
        KernelKMeans_model.fit(X_with_contamination)
        mean_KernelKMeans = kernelMetrics.pre_image(X_with_contamination, KernelKMeans_model.labels_, 
                         KernelKMeans_model.gamma, n_clusters, num_iter_conv = 200)
        
        bias_kmeans, bias_kernelkmeans[:,i] = kernelMetrics.calculate_bias(bias_kmeans,bias_kernelkmeans[:,i],
                                                        mean_KMeans,mean_KMeans_initial,
                                                        mean_KernelKMeans,mean_KernelKMeans_initial,
                                                        n_clusters,epoch)
    if epoch % 10 == 0:
        gaussianExperiment.show_graph_3d(X,y,X_contamination_epoch,int(epochs/10))

for i in range(1,5):
    kernelMetrics.create_graphs(bias_kmeans,bias_kernelkmeans[:,i],i, 5)