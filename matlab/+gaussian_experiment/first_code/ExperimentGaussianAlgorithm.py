# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:03:49 2014

@author: Alejandro

Definition: In this file we compare the k-means algorithm vs kernel k-means algorithms

We first generates data for multivariate normal with the class Robust, after 
that we generate random points who are considerated like contamination.


"""
import numpy as np
from sklearn.cluster import KMeans
from kkmeans import KernelKMeans
from sklearn.cross_validation import KFold
from random import sample
from FunctionRobust import *

    
    
# Load the data
data_total = np.load('DataGaussian/dataExperiment1.npz')
# Assign the data to the variable
data = data_total['data'] 
#Assign
contamination_full = data_total['contamination']


num_datos = 100
numiter = 10
num_cluster= 3
num_iter_conv = 100
num_outliers_grafica= 6
numoutliers = 150

sum_kmeans = np.zeros(num_outliers_grafica-5)
sum_kernelkmeans=np.zeros(num_outliers_grafica-5)
sesgo_kernelkmeans=np.zeros(num_outliers_grafica-5)
sesgo_kmeans = np.zeros(num_outliers_grafica-5)
par_gamma = np.zeros((num_outliers_grafica-5,numiter))


'''
    We use mean1 to mean3 variables to represents the mean of the data,
    this is load with data above.
'''
mean1  = data_total['mean'][0]
mean2  = data_total['mean'][1]
mean3  = data_total['mean'][2]
print mean1
print mean2
print mean3

for index,outliers in zip(xrange(0,num_outliers_grafica-5),xrange(5,num_outliers_grafica)):
    '''
    We use the outliers variable to count how many points on the graph are outliers
    we inicilize the variable outliers in 5 becuause we need this to allow us 
    to make cross-validation with five folds
    '''
    print index
    contamination = contamination_full[sample(range(num_outliers_grafica), outliers)]
    
    for iterations in xrange(numiter):
        '''
        We use cross validation aproach to calculate the systematic error.
        We compare 
        '''
        kf_data = KFold(len(data), n_folds=5, indices=True, shuffle=True)
        kf_contamination = KFold(len(contamination), n_folds=5, indices=True, shuffle=True)        
        
        for (train_data, test_data ),(train_contamination,test_contamination) in zip(kf_data,kf_contamination):
            '''
            Generate data, with cross-validation we use to train the kmeans algorithms and kernelkmeans.
            '''
            X = np.concatenate((data[train_data],contamination[train_contamination]),axis=0)
            X_without_contamination = data[train_data]
            X_test = np.concatenate((data[test_data],contamination[test_contamination]),axis=0)
            X_without_contamination_test = data[test_data]
            
            #With
            k_means_model = KMeans(n_clusters = 3, max_iter = 10000, random_state = 0)
            vector = k_means_model.fit_predict(X)
            predict = k_means_model.predict(X_without_contamination_test)
            error_kmeans = get_minimum_score(predict,test_data)    
            sum_kmeans[index] = np.sum(double(error_kmeans)) + sum_kmeans[index]


            [kernelKMeans_model,  gamma] = get_kkmeans_model(X, X_without_contamination, train_data)            
            par_gamma[index][iterations] = gamma 
            predict = kernelKMeans_model.predict(X_without_contamination_test)    
            error_kernelkmeans = get_minimum_score(predict,test_data)
            sum_kernelkmeans[index] = np.sum(double(error_kernelkmeans)) + sum_kernelkmeans[index]
            
            #with this we calculate the kernelkmeans            
            predict = kernelKMeans_model.predict(X)            
            z = pre_image(X, predict, gamma, num_cluster, num_iter_conv)
            for i in range(num_cluster):
                sesgo_kmeans[index] = np.abs(np.min((np.sum(k_means_model.cluster_centers_[i]-mean1),np.sum(k_means_model.cluster_centers_[i]-mean2),np.sum(k_means_model.cluster_centers_[i]-mean3))))+sesgo_kmeans[index]

            for i in range(num_cluster):
                sesgo_kernelkmeans[index] = np.abs(np.min((np.sum(z[i]-mean1),np.sum(z[i]-mean2),np.sum(z[i]-mean3))))+sesgo_kernelkmeans[index]
                
    sesgo_kmeans[index] = sesgo_kmeans[index] / numiter/5    
    sesgo_kernelkmeans[index] = sesgo_kernelkmeans[index]/numiter/5
    sum_kmeans[index] = sum_kmeans[index]/numiter/5
    sum_kernelkmeans[index] = sum_kernelkmeans[index]/numiter/5

np.savez('DataGaussian/ResultsTestKKMeanGaussianExperiment1', sesgo_kmeans = sesgo_kmeans, sesgo_kernelkmeans = sesgo_kernelkmeans,sum_kmeans = sum_kmeans,sum_kernelkmeans = sum_kernelkmeans, par_gamma = par_gamma)
