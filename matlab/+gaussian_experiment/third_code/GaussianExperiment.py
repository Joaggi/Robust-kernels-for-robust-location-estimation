# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:39:18 2014

@author: Alejandro
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from random import shuffle
from numpy.random import uniform
from math import sqrt
from sklearn.metrics import pairwise as pw
from random import sample
from sklearn.cluster import k_means
from KernelKMeans import KernelKMeans


class GaussianExperiment(object):
    def __init__(self, n_samples=1, n_outliers=1, n_clusters=1, n_features=1, n_experiment=1):
        self.n_samples = n_samples
        self.n_outliers = n_outliers        
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_experiment = n_experiment
        self.fig = plt.figure()
        self.aux_fig = 0
        
    def show_graph_3d(self, X,y,X_contaminated, n_subplots):
        self.aux_fig+=1
        dimension =  int(sqrt(n_subplots))+1
        self.ax = self.fig.add_subplot(dimension,dimension,self.aux_fig, projection='3d')
        self.m = 'o'
        self.ax.scatter(X[:,0], X[:,1], X[:,2], c=y, marker=self.m)
        self.ax.scatter(X_contaminated[:,0], X_contaminated[:,1], X_contaminated[:,2], \
            c=['g' for i in range(X_contaminated[:,0].shape[0])], marker='^')
        self.ax.set_xlabel('First Dimension')
        self.ax.set_ylabel('Second Dimension')
        self.ax.set_zlabel('Third Dimension')
        self.ax.set_title('Numbers of outliers: ' + str(X_contaminated.shape[0]))
        plt.show()
    
    def generate_data(self):
        if self.n_experiment == 1:
            self.X, self.y = make_blobs(n_samples=300, centers=self.n_clusters, n_features=3,random_state=3)
        if self.n_experiment == 2:
            self.X, self.y = make_blobs(n_samples=300, centers=self.n_clusters, n_features=3,random_state=5)
        return self.X,self.y
    
    def generate_contamination(self):
        if self.n_experiment == 1:
            self.X_contamination_total = np.zeros((self.n_outliers,self.n_features))
            for i in range (self.n_outliers):
                list_partition = [x for x in range(self.n_features)]
                shuffle(list_partition)
                dimension = np.zeros((3))
                dimension[0] = uniform(low=-10,high=-4,size=1)
                dimension[1] = uniform(low=-4,high=4,size=1)
                dimension[2] = uniform(low=4,high=10,size=1)
                self.X_contamination_total[i,:] = np.array(
                    ([dimension[list_partition[0]],dimension[list_partition[1]],
                      dimension[list_partition[2]]]))        
            return self.X_contamination_total
            
        if self.n_experiment == 2:
            self.X_contamination_total = np.zeros((self.n_outliers,self.n_features))
            for i in range (self.n_outliers):
                list_partition = [x for x in range(self.n_features)]
                shuffle(list_partition)
                dimension = np.zeros((3))
                dimension[0] = uniform(low=-10,high=0,size=1)
                dimension[1] = uniform(low=-10,high=5,size=1)
                dimension[2] = uniform(low=-3,high=10,size=1)
                self.X_contamination_total[i,:] = np.array(([dimension[list_partition[0]],
                    dimension[list_partition[1]],dimension[list_partition[2]]]))        
            return self.X_contamination_total   
            
    def save_data(self):
        np.savez('Data/data_experiment_' + str(self.n_experiment), X = self.X,
            labels = self.y, X_contamination = self.X_contamination_total,
            n_samples = self.n_samples ,  n_outliers = self.n_outliers,     
            n_features = self.n_features, n_clusters = self.n_clusters,
            n_experiment = self.n_experiment)
            
            
class KernelMetrics(object):
    
    def __init__(self):
        self.fig = plt.figure()
        self.aux_fig = 0
    
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
    
    def load_data(self,n_experiment):
        # Load the data
        data_total = np.load('Data/data_experiment_' + str(n_experiment) + '.npz')
        # Assign the data to the variable
        X = data_total['X'] 
        #Assign
        X_contamination = data_total['X_contamination']
        
        y = data_total['labels']
        
        n_samples = data_total['n_samples']
        n_outliers = data_total['n_outliers']      
        n_features = data_total['n_features']
        n_clusters = data_total['n_clusters']
        
        return X ,y,X_contamination , n_samples, n_outliers, n_features , n_clusters
    
    def pre_image(self,X,labels, gamma , n_clusters,num_iter_conv):
        z = np.zeros((4,3))
        clusters_indices = []
        for i in xrange(n_clusters):
            clusters_indices.append([])
        for j in xrange(n_clusters):
            clusters_indices[j] = np.where(labels == j)
        for i in xrange(n_clusters):
            z[i] = np.sum(X[clusters_indices[i]],axis=0)/X[clusters_indices[i]].shape[0]
        for i in xrange(num_iter_conv):
            for j in xrange(n_clusters):
                kernel = self.rbf_kernel(z[j],X[clusters_indices[j]],gamma)         
                for k in xrange(X.shape[1]):
                    z[j,k] = np.dot(X[clusters_indices[j],k],kernel)[0][0] / np.sum(kernel)
        return z
    
    def insert_contamination(self,X,X_contamination,epoch,n_outliers):
        X_contamination_epoch = X_contamination[sample(range(n_outliers), epoch)]
        #print X_contamination_epoch
        X_with_contamination = np.concatenate((X,X_contamination_epoch),axis=0)
        #print X_with_contamination
        return X_with_contamination, X_contamination_epoch
    
    def initial_means(self,X,n_clusters):
        mean_KMeans_initial,label, intertia = k_means(X, n_clusters, random_state = 1)
        
        KernelKMeans_model = KernelKMeans(n_clusters=n_clusters, random_state=1,
                             kernel="rbf", gamma=None, coef0=1,
                             verbose=0)        
        KernelKMeans_model.fit(X)
        mean_KernelKMeans_initial = self.pre_image(X, KernelKMeans_model.labels_, KernelKMeans_model.gamma, n_clusters, 100)
        return mean_KMeans_initial , mean_KernelKMeans_initial 
        
    def create_graphs(self,bias_kmeans,bias_kernelkmeans,n_subplot,n_gamma):
        self.aux_fig+=1
        dimension =  int(sqrt(n_gamma))+1
        self.ax = self.fig.add_subplot(dimension,dimension,self.aux_fig)
        self.ax.plot(range(bias_kmeans.shape[0]),bias_kmeans,label = 'K-Means')
        self.ax.plot(range(bias_kernelkmeans.shape[0]),bias_kernelkmeans,label = 'Kernel K Means')
        self.ax.set_xlabel('Numbers of Outliers')
        self.ax.set_ylabel('Bias')
        self.ax.set_title('Bias Gamma: 2^-(' + str(n_subplot) + ')')
    
    def calculate_bias(self,bias_kmeans,bias_kernelkmeans,mean_KMeans,mean_KMeans_initial,mean_KernelKMeans,mean_KernelKMeans_initial,n_clusters,epoch):
        for i in range(n_clusters):
            aux = 999999999999
            for j in range(n_clusters):
                if np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j])) < aux:
                    aux = np.abs(np.sum(mean_KMeans[i]-mean_KMeans_initial[j]))
            bias_kmeans[epoch] = aux + bias_kmeans[epoch]
    
    
        for i in range(n_clusters):
            aux = 999999999999
            for j in range(n_clusters):
                if np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j])) < aux:
                    aux = np.abs(np.sum(mean_KernelKMeans[i]-mean_KernelKMeans_initial[j]))
            bias_kernelkmeans[epoch] = aux+bias_kernelkmeans[epoch]
            
        return bias_kmeans, bias_kernelkmeans