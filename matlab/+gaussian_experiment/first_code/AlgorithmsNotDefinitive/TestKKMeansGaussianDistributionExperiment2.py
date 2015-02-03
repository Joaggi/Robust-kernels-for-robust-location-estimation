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
from kkmeans import *
from sklearn.cross_validation import KFold
from random import sample
from sklearn.metrics import pairwise as pw

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
        
def get_minimum_score(predict,test_data):
    '''
    This function allow us to get what is the real value of the label that is assign with 
    the kmeans algorithms.
    '''
    num_datos_cluster1 = np.sum(np.where(test_data[np.where(test_data>=0)] <100)[0].shape)
    num_datos_cluster2 = np.sum(np.where(test_data[np.where(test_data>=100)] <200)[0].shape)
    num_datos_cluster3 = np.sum(np.where(test_data[np.where(test_data>=200)] <300)[0].shape)
    
    segundo1 = np.empty(num_datos_cluster1)
    segundo2 = np.empty(num_datos_cluster2)
    segundo3 = np.empty(num_datos_cluster3)
    segundo1[:] = 2
    segundo2[:] = 2
    segundo3[:] = 2

    vector1 = np.concatenate((np.zeros(num_datos_cluster1),np.ones(num_datos_cluster2),segundo3),axis=0)
    vector2 = np.concatenate((np.zeros(num_datos_cluster1),segundo2,np.ones(num_datos_cluster3)),axis=0)
    vector3 = np.concatenate((np.ones(num_datos_cluster1),np.zeros(num_datos_cluster2),segundo3),axis=0)
    vector4 = np.concatenate((np.ones(num_datos_cluster1),segundo2,np.zeros(num_datos_cluster3)),axis=0)
    vector5 = np.concatenate((segundo1,np.zeros(num_datos_cluster2),np.ones(num_datos_cluster3)),axis=0)
    vector6 = np.concatenate((segundo1,np.ones(num_datos_cluster2),np.zeros(num_datos_cluster3)),axis=0)

    error1 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector1)]))
    error2 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector2)]))
    error3 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector3)]))
    error4 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector4)]))
    error5 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector5)]))
    error6 = np.sum(np.array([1 if x!=y else 0 for (x,y) in zip(predict,vector6)]))

    error = min(error1,error2,error3,error4,error5,error6)
    return error
    
    
# Load the data
data_total = np.load('DataGaussian/dataExperiment2.npz')
# Assign the data to the variable
data = data_total['data'] 
#Assian
contamination_full = data_total['contamination']


num_datos = 100
numiter = 50
ncluster= 3
numiter_conv = 1000
num_cluster = np.array([])


sum_kmeans = []
sum_kernelkmeans=[]
sesgo_kernelkmeans= []
sesgo_kmeans = []
par_gamma = []

num_outliers_grafica= 150

'''
    We use mean1 to mean3 variables to represents the mean of the data,
    this is load with data above.
'''
mean1  = data_total['mean'][0]
mean2  = data_total['mean'][1]
mean3  = data_total['mean'][2]


for index,outliers in zip(xrange(0,num_outliers_grafica-5),xrange(5,num_outliers_grafica)):
    '''
    We use the outliers variable to count how many points on the graph are outliers
    we inicilize the variable outliers in 5 becuause we need this to allow us 
    to make cross-validation with five folds
    '''
    print index
    numoutliers = 150
    sum_kmeans.append([0])
    sum_kernelkmeans.append([0])
    sesgo_kernelkmeans.append([0])
    sesgo_kmeans.append([0])
    par_gamma.append([])
    contamination = contamination_full[sample(range(num_outliers_grafica), outliers)]
    
    for iterations in xrange(numiter):
        '''
        We use cross validation aproach to calculate the systematic error.
        We compare 
        '''
        par_gamma[index].append(0)
        kf_data = KFold(len(data), n_folds=5, indices=True)
        kf_contamination = KFold(len(contamination), n_folds=5, indices=True)        
        
        for (train_data, test_data ),(train_contamination,test_contamination) in zip(kf_data,kf_contamination):
            
            X = np.concatenate((data[train_data],contamination[train_contamination]),axis=0)
            X_without_contamination = data[train_data]
            X_test = np.concatenate((data[test_data],contamination[test_contamination]),axis=0)
            X_without_contamination_test = data[test_data]
            
            k_means = KMeans(n_clusters = 3, max_iter = 10000, random_state = 0)
            vector = k_means.fit_predict(X)
            predict = k_means.predict(X_without_contamination_test)
            
            error = get_minimum_score(predict,test_data)    
            
            sum_kmeans[index] = np.sum(error) + sum_kmeans[index]
            error_2 = 1000000
            
            for i in xrange(18):
                try:
                    km = KernelKMeans(n_clusters=3, max_iter=1000, verbose=0,kernel='rbf',gamma=2**-i)
                    km.fit_predict(X)
                    predict = km.predict(X_without_contamination)
                    error = get_minimum_score(predict,train_data)
                    if error < error_2 : 
                        error_2 = error
                        kmmean = km
                        gamma = 2**-i
                        par_gamma[index][iterations] = 2**-i
                except:
                    pass
                
            predict = kmmean.predict(X_without_contamination_test)    
            error_2 = get_minimum_score(predict,test_data)
            sum_kernelkmeans[index] = np.sum(error_2) + sum_kernelkmeans[index]
            
            z = np.random.uniform(low=-30,high=30,size=X.shape[1]).reshape((1,X.shape[1]))
            for i in xrange(ncluster-1):
                z = np.concatenate((z,np.random.uniform(low=-30,high=30,size=X.shape[1]).reshape((1,X.shape[1]))),axis=0)
            predict = kmmean.predict(X)
            clusters_indices = []
            for i in xrange(ncluster):
                clusters_indices.append([])
            for j in xrange(ncluster):
                clusters_indices[j] = np.where(predict == j)
            
            for i in xrange(numiter_conv):
                for j in xrange(ncluster):
                    kernel = rbf_kernel(z[j],X[clusters_indices[j]],gamma)         
                    for k in xrange(X.shape[1]):
                        z[j,k] = np.dot(np.transpose(kernel),np.transpose(X[clusters_indices[j],k])) / np.sum(kernel)

            for i in range(ncluster):
                sesgo_kmeans[index] = np.abs(np.min((np.sum(k_means.cluster_centers_[i]-mean1),np.sum(k_means.cluster_centers_[i]-mean2),np.sum(k_means.cluster_centers_[i]-mean3))))+sesgo_kmeans[index]
                    
            
            for i in range(ncluster):
                sesgo_kernelkmeans[index] = np.abs(np.min((np.sum(z[i]-mean1),np.sum(z[i]-mean2),np.sum(z[i]-mean3))))+sesgo_kernelkmeans[index]
                
    sesgo_kmeans[index] = sesgo_kmeans[index] / numiter/5    
    sesgo_kernelkmeans[index] = sesgo_kernelkmeans[index]/numiter/5
    sum_kmeans[index] = sum_kmeans[index]/numiter/5
    sum_kernelkmeans[index] = sum_kernelkmeans[index]/numiter/5
    
print sum_kmeans
print sum_kernelkmeans




import pylab as plt
# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(range(5,num_outliers_grafica),sesgo_kmeans)
axarr[0].plot(range(5,num_outliers_grafica),sesgo_kernelkmeans)
axarr[0].set_title('Sharing X axis')
axarr[1].plot(range(5,num_outliers_grafica),sum_kmeans)
axarr[1].plot(range(5,num_outliers_grafica),sum_kernelkmeans)
axarr[2].plot(range(5,num_outliers_grafica),par_gamma)
plt.show()




plt.clf()
plt.plot(range(5,num_outliers_grafica),sesgo_kmeans)
plt.plot(range(5,num_outliers_grafica),sesgo_kernelkmeans)


plt.clf()
plt.plot(range(5,num_outliers_grafica),sum_kmeans)
plt.plot(range(5,num_outliers_grafica),sum_kernelkmeans)

np.savez('DataGaussian/resultTestKKMeanGaussian2', sesgo_kmeans, sesgo_kernelkmeans,sum_kmeans,sum_kernelkmeans)