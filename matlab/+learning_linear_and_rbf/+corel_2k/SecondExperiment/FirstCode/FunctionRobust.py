# -*- coding: utf-8 -*-
"""
Created on Thu May 08 18:22:03 2014

@author: Alejandro
"""

import numpy as np
from sklearn.metrics import pairwise as pw
from kkmeans import KernelKMeans
import matplotlib.pyplot as plt

def rbf_kernel(Z, X, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-γ ||x-y||²)

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = pw.euclidean_distances(X, Z, squared=True)
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K
        
def get_minimum_score(predict,test_data,num_clusters):
    '''
    This function allow us to get what is the real value of the label that is assign with 
    the kmeans algorithms.
    '''
    aux = 0
    clasificiacion  = 0
    for i in range(num_clusters):
        #print predict.shape
        #print test_data.shape
        #print test_data[np.where(test_data[np.where(test_data>=i*100)] <i*100+100)[0]]
        #print predict
        #print test_data
        #print predict[aux:aux + test_data[np.where(test_data[np.where(test_data>=i*100)] <i*100+100)[0]].shape[0]]
        cluster = predict[aux:aux + test_data[np.where(test_data[np.where(test_data>=i*100)] <i*100+100)[0]].shape[0]]
        aux = test_data[np.where(test_data[np.where(test_data>=i*100)] <i*100+100)[0]].shape[0]+ aux
        counts = np.bincount(cluster)
        #print counts
        #print counts[np.argmax(counts)]
        clasificiacion = clasificiacion + counts[np.argmax(counts)]
        #raw_input('Ingrese')
    return float(clasificiacion) / float(predict.shape[0])
    
    
def pre_image(X,predict, gamma , num_cluster,num_iter_conv):
    z = np.zeros((3,4))
    clusters_indices = []
    for i in xrange(num_cluster):
        clusters_indices.append([])
    for j in xrange(num_cluster):
        clusters_indices[j] = np.where(predict == j)
    for i in xrange(num_cluster):
        z[i] = np.sum(X[clusters_indices[i]],axis=0)/X[clusters_indices[i]].shape[0]
    for i in xrange(num_iter_conv):
        for j in xrange(num_cluster):
            kernel = rbf_kernel(z[j],X[clusters_indices[j]],gamma)         
            for k in xrange(X.shape[1]):
                z[j,k] = np.dot(X[clusters_indices[j],k],kernel)[0][0] / np.sum(kernel)
    return z
    
def get_kkmeans_model(X, X_without_contamination, train_data,num_clusters):
    error_2 = 1000000
    for i in xrange(1,15):
        try:

            kernelKMeans = KernelKMeans(n_clusters=num_clusters, max_iter=1000, verbose=0,kernel='rbf',gamma=2**-i)
            kernelKMeans.fit(X)
            predict = kernelKMeans.predict(X_without_contamination)
            #print predict
            error = get_minimum_score(predict,train_data,num_clusters)
            #print error
            if error < error_2 : 
                error_2 = error
                kernelKMeans_model = kernelKMeans
                gamma = 2**-i   
        except:
            pass
    kernelKMeans_model.predict(X_without_contamination)
    #print gamma
    #raw_input('ingrese')
    return kernelKMeans_model,  gamma

def create_graphs_without_gamma(url_file):
    data = np.load('DataGaussian/' + url_file + '.npz')
    plt.subplot(2, 1, 1)
    plt.plot(range(data['arr_0'].shape[0]), data['arr_0'],label = 'K-Means')
    plt.plot(range(data['arr_1'].shape[0]), data['arr_1'],label = 'Kernel K Means')
    plt.ylabel('Systematic error with pre-image')
    plt.title('Sharing Y axis')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(data['arr_2'].shape[0]), data['arr_2'],label = 'K-Means')
    plt.plot(range(data['arr_3'].shape[0]), data['arr_3'],label = 'Kernel K Means')
    plt.ylabel('Number of missclassified points')
    plt.legend()
    
    plt.show()
    

def create_graphs_with_gamma(url_file):
    data = np.load('DataGaussian/' + url_file + '.npz')
    plt.subplot(3, 1, 1)
    plt.plot(range(data['sesgo_kmeans'].shape[0]), data['sesgo_kmeans'],label = 'K-Means')
    plt.plot(range(data['sesgo_kernelkmeans'].shape[0]), data['sesgo_kernelkmeans'],label = 'Kernel K Means')
    plt.ylabel('Systematic error with pre-image')
    plt.title('Sharing Y axis')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(range(data['sum_kmeans'].shape[0]), data['sum_kmeans'],label = 'K-Means')
    plt.plot(range(data['sum_kernelkmeans'].shape[0]), data['sum_kernelkmeans'],label = 'Kernel K Means')
    plt.ylabel('Number of missclassified points')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(range(data['par_gamma'].shape[0]), data['par_gamma'])
    plt.xlabel('Number of contaminated points')
    plt.ylabel('Parammeter Gamma')
    
    plt.show()
    
def create_graphs_with_gamma_without_sesgo(url_file):
    data = np.load('Results/' + url_file + '.npz')
    print data['sum_kmeans']
    print data['sum_kernelkmeans']
    print data['par_gamma']
    plt.subplot(2, 1, 1)
    plt.plot(range(data['sum_kmeans'].shape[0]), data['sum_kmeans'],label = 'K-Means')
    plt.plot(range(data['sum_kernelkmeans'].shape[0]), data['sum_kernelkmeans'],label = 'Kernel K Means')
    plt.ylabel('Number of missclassified points')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(data['par_gamma'].shape[0]), data['par_gamma'])
    plt.xlabel('Number of contaminated points')
    plt.ylabel('Parammeter Gamma')
    
    plt.show()
    