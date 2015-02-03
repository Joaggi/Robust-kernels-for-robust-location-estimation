# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:39:18 2014

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import pairwise as pw
from random import sample
from sklearn.cluster import k_means
from KernelKMeans import KernelKMeans
from scipy import io
from sklearn.preprocessing import normalize
from scipy.stats import mode
         
class KernelMetrics(object):
    
    def __init__(self, n_clusters, n_clusters_outliers , n_experiment, n_iter = 10):
        self.n_clusters = n_clusters
        self.n_outliers = n_clusters_outliers * 100
        self.n_samples = n_clusters* 100
        self.n_clusters_outliers = n_clusters_outliers
        self.n_experiment = n_experiment
        self.n_iter = n_iter
        
    def init_figure(self):
        self.aux_fig_bias = 0
        self.aux_fig_purity = 0
        self.aux_fig_entropy = 0
        self.fig_bias = plt.figure()
        self.fig_purity = plt.figure()
        self.fig_entropy = plt.figure()
        
    def defineVariablesCorel(self):
        self.bias = {}
        self.purity= {}
        self.entropy = {}
        self.calculateEntropy = {}
        self.model = {}   
        self.bias['kernelkmeans']=np.zeros((self.n_outliers+1,5,self.n_iter))
        self.bias['kmeans'] = np.zeros((self.n_outliers+1,self.n_iter))
        self.purity['kmeans'] = np.zeros((self.n_outliers+1,self.n_clusters,self.n_iter))
        self.purity['kernelkmeans'] = np.zeros((self.n_outliers+1,self.n_clusters,5,self.n_iter))
        self.entropy['kmeans'] = np.zeros((self.n_outliers+1,self.n_clusters,self.n_iter))
        self.calculateEntropy['kmeans'] = np.zeros((self.n_outliers+1,self.n_iter))
        self.entropy['KernelKMeans'] = np.zeros((self.n_outliers+1,self.n_clusters,5,self.n_iter))
        self.calculateEntropy['KernelKMeans'] = np.zeros((self.n_outliers+1,5,self.n_iter))    
        return self.bias , self.purity ,self.entropy, self.calculateEntropy, self.model

    
    def rbf_kernel(self,Z, X, gamma=None):
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
    
    def load_data(self):
        n_datos = self.n_clusters * 100
        n_outliers = self.n_clusters_outliers * 100
        
        # Load the data
        data_total = io.loadmat('Data/dat.mat')
        # Assign the data to the variable
        data_full = data_total['dat'] 
        
        data_full = normalize(data_full, norm='l2', axis= 1)
        
        X = np.transpose(data_full[:,0:n_datos])
        X_contamination = np.transpose(data_full[:,n_datos:n_datos+n_outliers])
        
        self.y = np.zeros(n_datos)
        aux = 0
        for i in range(n_datos):
            self.y[i] = aux
            if i %100 == 0:
                aux+=1
                
        self.n_samples = X.shape[0]
        n_features = X.shape[1]
        
        return X ,self.y,X_contamination , self.n_samples, self.n_outliers, n_features , self.n_clusters
    
    def pre_image(self,X,labels, gamma , num_iter_conv):
        z = np.zeros((self.n_clusters,X.shape[1]))
        clusters_indices = []
        for i in xrange(self.n_clusters):
            clusters_indices.append([])
        for j in xrange(self.n_clusters):
            clusters_indices[j] = np.where(labels == j)
        for i in xrange(self.n_clusters):
            z[i] = np.sum(X[clusters_indices[i]],axis=0)/X[clusters_indices[i]].shape[0]
        for i in xrange(num_iter_conv):
            for j in xrange(self.n_clusters):
                kernel = self.rbf_kernel(z[j],X[clusters_indices[j]],gamma)     
                for k in xrange(X.shape[1]):
                    z[j,k] = np.dot(X[clusters_indices[j],k],kernel)[0][0] / np.sum(kernel)
        return z
    
    def insert_contamination(self,X,X_contamination,epoch):
        X_contamination_epoch = X_contamination[sample(range(self.n_outliers), epoch)]
        #print X_contamination_epoch
        X_with_contamination = np.concatenate((X,X_contamination_epoch),axis=0)
        #print X_with_contamination
        return X_with_contamination, X_contamination_epoch
    
    def initial_means(self,X, gamma = None):
        mean_KMeans_initial,KMeans_label, intertia = k_means(X, self.n_clusters, random_state = 1)
        
        KernelKMeans_model = KernelKMeans(n_clusters=self.n_clusters,
                             kernel="rbf", gamma=gamma, coef0=1,
                             verbose=0,max_iter=200)        
        KernelKMeans_model.fit(X)
        mean_KernelKMeans_initial = self.pre_image(X, 
                             KernelKMeans_model.labels_, KernelKMeans_model.gamma, 100)
        return mean_KMeans_initial , mean_KernelKMeans_initial , KMeans_label , KernelKMeans_model.labels_ 
        
    def create_graphs_bias(self,n_subplot,n_gamma):
        self.aux_fig_bias+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_bias.add_subplot(dimension,dimension,self.aux_fig_bias)
        self.ax.plot(range(self.bias['kmeans'].shape[0]),self.bias['kmeans'],label = 'K-Means')
        self.ax.plot(range(self.bias['kernelkmeans'][:,n_subplot].shape[0]),self.bias['kernelkmeans'][:,n_subplot],label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('Bias')
        self.ax.set_title('Bias Gamma: 2^-(' + str(n_subplot) + ')')
        
    def create_graphs_purity(self,n_subplot,n_gamma):
        self.aux_fig_purity+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_purity.add_subplot(dimension,dimension,self.aux_fig_purity)
        self.ax.plot(range(self.purity['kmeans'].shape[0]),self.purity['kmeans'],label = 'K-Means')
        self.ax.plot(range(self.bias['kernelkmeans'].shape[0]),self.bias['kernelkmeans'],label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('purity')
        self.ax.set_title('purity Gamma: 2^-(' + str(n_subplot) + ')')
        
    def create_graphs_entropy(self,n_subplot,n_gamma):
        self.aux_fig_entropy+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_entropy.add_subplot(dimension,dimension,self.aux_fig_entropy)
        self.ax.plot(range(self.entropy['kmeans'].shape[0]),self.entropy['kmeans'],label = 'K-Means')
        self.ax.plot(range(self.entropy['KernelKMeans'].shape[0]),self.entropy['KernelKMeans'],label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('entropy')
        self.ax.set_title('entropy Gamma: 2^-(' + str(n_subplot) + ')')
    
    def calculate_bias(self,mean_KMeans,mean_KMeans_initial,mean_KernelKMeans,mean_KernelKMeans_initial,epoch,n_iter=0):
        for i in range(self.n_clusters):
            aux = -1
            for j in range(self.n_clusters):
                if np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j])) < aux or aux ==-1:
                    aux = np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j]))
            self.bias['kmeans'][epoch,n_iter] = aux + self.bias['kmeans'][epoch,n_iter]
    
        for i in range(self.n_clusters):
            aux = -1
            for j in range(self.n_clusters):
                if np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j])) < aux or aux ==-1:
                    aux = np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j]))
            self.bias['kernelkmeans'][epoch,n_iter] = aux+self.bias['kernelkmeans'][epoch,n_iter]
        return self.bias['kmeans'], self.bias['kernelkmeans']
        
    def calculate_purity(self, labels,epoch,n_iter=0,gamma=None):
        labels = labels[0:self.n_samples]
        if gamma== None:
            for i in range(self.n_clusters):
                self.purity['kmeans'][epoch,i,n_iter] = \
                    (mode(self.y[np.where(labels == i)])[1]/np.where(labels == i)[0].shape[0])[0]
            return self.purity['kmeans']
        else:
            for i in range(self.n_clusters):
                self.purity['kernelkmeans'][epoch,i,gamma,n_iter] =\
                    (mode(self.y[np.where(labels == i)])[1]/np.where(labels == i)[0].shape[0])[0]
            return self.purity['kernelkmeans'] 
            
    def calculate_entropy(self,labels,epoch,n_iter=0,gamma=None):
        labels = labels[0:self.n_samples]
        #print self.n_samples
        if gamma==None:
            for i in range(self.n_clusters):
                #print float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
                self.entropy['kmeans'][epoch,i,n_iter] = float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
            self.calculateEntropy['kmeans'][epoch,n_iter] = \
                -np.dot(np.log(self.entropy['kmeans'][epoch,:,n_iter]),self.entropy['kmeans'][epoch,:,n_iter])
            return self.entropy['kmeans'], self.calculateEntropy['kmeans']
        else:
            for i in range(self.n_clusters):
                self.entropy['KernelKMeans'][epoch,i,gamma,n_iter] = float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
            self.calculateEntropy['KernelKMeans'][epoch,gamma,n_iter] =\
                -np.dot(np.log(self.entropy['KernelKMeans'][epoch,:,gamma,n_iter]),self.entropy['KernelKMeans'][epoch,:,gamma,n_iter])
            return self.entropy['KernelKMeans'], self.calculateEntropy['KernelKMeans']       
            