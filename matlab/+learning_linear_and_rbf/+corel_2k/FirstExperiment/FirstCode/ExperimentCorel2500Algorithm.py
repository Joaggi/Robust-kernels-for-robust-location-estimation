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
from sklearn.cross_validation import KFold
from random import sample
from FunctionRobust import *
from scipy import io



def experiment_corel(num_datos, num_outliers, num_clusters, num_experiment):

    if num_clusters ==0:
        num_clusters = int(num_datos /100)
        
    # Load the data
    data_total = io.loadmat('dat.mat')
    # Assign the data to the variable
    data_full = data_total['dat'] 
    data = np.transpose(data_full[:,0:num_datos])
    contamination_full = np.transpose(data_full[:,num_datos:num_datos+num_outliers])

   
    #print data.shape
    #print contamination_full.shape    
    
    numiter = 5
    num_outliers_grafica= 6
    
    sum_kmeans = np.zeros(num_outliers_grafica-5)
    sum_kernelkmeans=np.zeros(num_outliers_grafica-5)
    par_gamma = np.zeros((num_outliers_grafica-5,numiter))
    
    
    '''
        We use mean1 to mean3 variables to represents the mean of the data,
        this is load with data above.
    '''
    
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
                #X_test = np.concatenate((data[test_data],contamination[test_contamination]),axis=0)
                #X_without_contamination_test = data[test_data]
                
                #With
                k_means_model = KMeans(n_clusters = num_clusters, max_iter = 10000, random_state = 0)
                vector = k_means_model.fit_predict(X)
                
                predict = k_means_model.predict(X_without_contamination)
                error_kmeans = get_minimum_score(predict,train_data,num_clusters)    
                sum_kmeans[index] = np.sum(float(error_kmeans)) + sum_kmeans[index]
    
                [kernelKMeans_model,  gamma] = get_kkmeans_model(X, X_without_contamination, train_data, num_clusters)            
                par_gamma[index][iterations] = gamma 
                
                predict = kernelKMeans_model.predict(X_without_contamination)    
                #print kernelKMeans_model.get_params
                error_kernelkmeans = get_minimum_score(predict,train_data,num_clusters)
                sum_kernelkmeans[index] = np.sum(float(error_kernelkmeans)) + sum_kernelkmeans[index]
            
                    
        sum_kmeans[index] = sum_kmeans[index]/numiter/5
        sum_kernelkmeans[index] = sum_kernelkmeans[index]/numiter/5
    
    np.savez('Results/ResultsTestKKMeanCorelExperiment_' + str(num_experiment), sum_kmeans = sum_kmeans,sum_kernelkmeans = sum_kernelkmeans, par_gamma = par_gamma)


experiment_corel(num_datos=1000, num_outliers=300, num_clusters=10, num_experiment=1)
#experiment_corel(num_datos=1000, num_outliers=200, num_clusters=10, num_experiment=2)
#experiment_corel(num_datos=1000, num_outliers=300, num_clusters=10, num_experiment=3)
#experiment_corel(num_datos=2000, num_outliers=100, num_clusters=20, num_experiment=4)
#experiment_corel(num_datos=1000, num_outliers=200, num_clusters=20, num_experiment=5)
#experiment_corel(num_datos=1000, num_outliers=300, num_clusters=20, num_experiment=6)
#experiment_corel(num_datos=1000, num_outliers=500, num_clusters=20, num_experiment=7)