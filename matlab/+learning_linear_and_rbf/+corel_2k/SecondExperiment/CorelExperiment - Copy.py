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
from sklearn.cluster import KMeans
from KernelKMeans import KernelKMeans
from scipy import io
from sklearn.preprocessing import normalize
from scipy.stats import mode
         
class Metrics(object):
    
    def __init__(self, n_clusters, n_clusters_outliers , n_experiment):
        self.n_clusters = n_clusters
        self.n_outliers = n_clusters_outliers * 100
        self.n_samples = n_clusters* 100
        self.n_clusters_outliers = n_clusters_outliers
        self.n_experiment = n_experiment

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
    
    def load_data(self, norm='l2'):
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
            if i %100 == 0:
                aux+=1
            self.y[i] = aux
        
        self.n_samples = X.shape[0]
        n_features = X.shape[1]
        
        return X ,self.y,X_contamination , self.n_samples, self.n_outliers, n_features , self.n_clusters
    
    def pre_image(self,X,labels, gamma=None , num_iter_conv=50):
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
    
    def insert_contamination(self,X,X_contamination,pointsOfContaminations):
        X_contamination_epoch = X_contamination[sample(range(self.n_outliers), pointsOfContaminations)]
        X_with_contamination = np.concatenate((X,X_contamination_epoch),axis=0)
        return X_with_contamination, X_contamination_epoch
    
    def kmeans_model(self,X):
        model = KMeans()
        model.fit(X, self.n_clusters)
        mean_KMeans_initial = model.cluster_centers_
        KMeans_label = model.lables_
        return mean_KMeans_initial , KMeans_label
        
    def kernelkmeans_model(self,X,gamma=None):
        KernelKMeans_model = KernelKMeans(n_clusters=self.n_clusters, random_state=1,
                             kernel="rbf", gamma=gamma, coef0=1,
                             verbose=0)        
        KernelKMeans_model.fit(X)
        mean_KernelKMeans_initial = self.pre_image(X, 
                             KernelKMeans_model.labels_, KernelKMeans_model.gamma, 100)
        return  mean_KernelKMeans_initial , KernelKMeans_model.labels_
        
    def initial_means(self, X, gamma = None,model ='kmeans'):
        if model == 'kmeans':
            return self.kmeans_model(X)
        if model == 'kernelkmeans':
            return self.kernelkmeans_model
            
            
    def calculate_bias(self,bias,mean):
         for i in range(self.n_clusters):
            aux = -1
            for j in range(self.n_clusters):
                if np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j])) < aux or aux ==-1:
                    aux = np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j]))
            bias_kmeans[epoch] = aux + bias_kmeans[epoch]       

    def calculate_bias(self,bias_kmeans,bias_kernelkmeans,mean_KMeans,mean_KMeans_initial,mean_KernelKMeans,mean_KernelKMeans_initial,epoch):
        for i in range(self.n_clusters):
            aux = -1
            for j in range(self.n_clusters):
                if np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j])) < aux or aux ==-1:
                    aux = np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j]))
            bias_kmeans[epoch] = aux + bias_kmeans[epoch]
    
        for i in range(self.n_clusters):
            aux = -1
            for j in range(self.n_clusters):
                if np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j])) < aux or aux ==-1:
                    aux = np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j]))
            bias_kernelkmeans[epoch] = aux+bias_kernelkmeans[epoch]
        return bias_kmeans, bias_kernelkmeans
        
    def calculate_purity(self, vector_purity,labels,epoch,gamma=None):
        labels = labels[0:self.n_samples]
        if gamma== None:
            for i in range(self.n_clusters):
                vector_purity[epoch,i] = (mode(self.y[np.where(labels == i)])[1]/np.where(labels == i)[0].shape[0])[0]
            return vector_purity
        else:
            for i in range(self.n_clusters):
                vector_purity[epoch,i,gamma] = (mode(self.y[np.where(labels == i)])[1]/np.where(labels == i)[0].shape[0])[0]
            return vector_purity
            
    def calculate_entropy(self,vector_entropy,vector_entropy_calculate, labels,epoch,gamma=None):
        labels = labels[0:self.n_samples]
        #print self.n_samples
        if gamma==None:
            for i in range(self.n_clusters):
                #print float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
                vector_entropy[epoch,i] = float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
            vector_entropy_calculate[epoch] = -np.dot(np.log(vector_entropy[epoch,:]),vector_entropy[epoch,:])
            return vector_entropy, vector_entropy_calculate
        else:
            for i in range(self.n_clusters):
                vector_entropy[epoch,i,gamma] = float(np.where(labels==i)[0].shape[0])/float(self.n_samples)
            vector_entropy_calculate[epoch,gamma] = -np.dot(np.log(vector_entropy[epoch,:,gamma]),vector_entropy[epoch,:,gamma])
            return vector_entropy, vector_entropy_calculate       
            
class KernelCorelExperiment(object):

    def __init__(self, n_clusters, n_clusters_outliers , n_experiment):
        self.n_clusters = n_clusters
        self.n_outliers = n_clusters_outliers * 100
        self.n_samples = n_clusters* 100
        self.n_clusters_outliers = n_clusters_outliers
        self.n_experiment = n_experiment
        
    def defineVariablesCorel(self):
        self.bias = {}
        self.purity= {}
        self.entropy = {}
        self.calculate_entropy = {}
        self.model = {}   
        self.bias['kernelkmeans']=np.zeros((self.n_outliers+1,11))
        self.bias['kmeans'] = np.zeros(self.n_outliers+1)
        self.purity['kmeans'] = np.zeros((self.n_outliers+1,self.n_clusters))
        self.purity['kernelkmeans'] = np.zeros((self.n_outliers+1,self.n_clusters,5))
        self.entropy['kmeans'] = np.zeros((self.n_outliers+1,self.n_clusters))
        self.calculate_entropy['kmeans'] = np.zeros(self.n_outliers+1)
        self.entropy['KernelKMeans'] = np.zeros((self.n_outliers+1,self.n_clusters,5))
        self.calculate_entropy['KernelKMeans'] = np.zeros((self.n_outliers+1,5))    
        return self.bias , self.purity ,self.entropy, self.calculate_entropy, self.model
            
     
        
        
        
class CreateGraphRobustness(object):
    def init_figure(self):
        self.aux_fig_bias = 0
        self.aux_fig_purity = 0
        self.aux_fig_entropy = 0
        self.fig_bias = plt.figure()
        self.fig_purity = plt.figure()
        self.fig_entropy = plt.figure()
        
    def create_graphs_bias(self,bias_kmeans,bias_kernelkmeans,n_subplot,n_gamma):
        self.aux_fig_bias+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_bias.add_subplot(dimension,dimension,self.aux_fig_bias)
        self.ax.plot(range(bias_kmeans.shape[0]),bias_kmeans,label = 'K-Means')
        self.ax.plot(range(bias_kernelkmeans.shape[0]),bias_kernelkmeans,label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('Bias')
        self.ax.set_title('Bias Gamma: 2^-(' + str(n_subplot) + ')')
        
    def create_graphs_purity(self,purity_kmeans,purity_kernelkmeans,n_subplot,n_gamma):
        self.aux_fig_purity+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_purity.add_subplot(dimension,dimension,self.aux_fig_purity)
        self.ax.plot(range(purity_kmeans.shape[0]),purity_kmeans,label = 'K-Means')
        self.ax.plot(range(purity_kernelkmeans.shape[0]),purity_kernelkmeans,label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('purity')
        self.ax.set_title('purity Gamma: 2^-(' + str(n_subplot) + ')')
        
    def create_graphs_entropy(self,entropy_kmeans,entropy_kernelkmeans,n_subplot,n_gamma):
        self.aux_fig_entropy+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig_entropy.add_subplot(dimension,dimension,self.aux_fig_entropy)
        self.ax.plot(range(entropy_kmeans.shape[0]),entropy_kmeans,label = 'K-Means')
        self.ax.plot(range(entropy_kernelkmeans.shape[0]),entropy_kernelkmeans,label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('entropy')
        self.ax.set_title('entropy Gamma: 2^-(' + str(n_subplot) + ')')
                