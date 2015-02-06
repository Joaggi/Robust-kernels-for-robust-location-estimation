# -*- coding: utf-8 -*-
"""
Created on Sat May 03 17:42:17 2014

@author: Alejandro
"""

from scipy import random
import numpy as np
from numpy.random import multivariate_normal
from random import shuffle
from sklearn.metrics import pairwise as pw

class Robust(object):
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
    
    def random_mean_and_matrix_semidefinite(self,random_means,random_covariances,array = True, num_points = 1):
        ''' 
            Definition:     
                
            Parameters
            ----------    
            random_means: Matrix 
                    Have in each row a two column vector [(low = a,high = b)]
                Example:
                    means = [[0,1],[2,3],[6,7]]
                    Where each row indicates in which range the uniform random generator can takes values
            
            random_covariances: Matrix
                    Is a square matrix, where each row has a column vector, who has inside a vector with two components
                Example:
                    A matrix 3x3.
                    covariance = [[[a11,b11],[a12,b12],[a13,b13]],[[a21,b21],[a22,b22],[a23,b23]],[[a31,b31],[a32,b32],[a33,b33]]]
                    Where each row indicates in which range the uniform random generator can takes values
                    a = [[[1,2],[-5,5],[-100,6]],[[135,683],[2,285],[-135,13]],[[58,135],[16,35],[5,68478]]]
        '''
        mean = []
        for random_mean in random_means:        
            mean.append(random.uniform(low=random_mean[0],high=random_mean[1]))
        
        if array == True:
            covariance = random.rand( random_means.shape[0] , random_means.shape[0])*(random_covariances[1] - random_covariances[0]) + random_covariances[0]
            covariance = covariance * covariance.transpose()
        else:
            covariance = []
            for random_covariance in random_covariances:
                covariance.append([])
                for points in random_covariance:
                    covariance[-1].append(random.uniform(low=points[0],high=points[1]))
            covariance = np.array(covariance)
            #A*A' is a semedefinite matrix 
            covariance = covariance*covariance.transpose()
        
        
        return  [mean,multivariate_normal(mean,covariance,num_points)]
    
    def save_data_generate(self,data_generate):
        np.savez('DataGaussian/data', vector = data_generate)
       
    def data_generator(self,num_gaussian=3,num_points= 100):
        '''
        
        '''
        puntos_media1  = np.array([[-5,-4],[-2,-1],[1,2],[4,5]])
        puntos_media2  = np.array([[-2,-1],[-5,-4],[-5,-4],[1,2]])
        puntos_media3  = np.array([[1,2],[-5,-4],[4,5],[1,5]])
        
        #covarianza1 = np.array([[[1, 2], [-5, 5], [-100, 6]], [[135, 683], [2, 285], [-135, 13]], [[58, 135], [16, 35], [5, 68478]]])
        #covarianza2 = np.array([[[135, 683], [2, 285], [-135, 13]],[[1, 2], [-5, 5], [-100, 6]], [[58, 135], [16, 35], [5, 68478]]])
        #covarianza3 = np.array([[[1, 2], [-5, 5], [-100, 6]], [[58, 135], [16, 35], [5, 68478]], [[135, 683], [2, 285], [-135, 13]]])

        data = self.random_mean_and_matrix_semidefinite(puntos_media1, np.array([-2,2]),array = True , num_points = 100)
        data = np.append(data,self.random_mean_and_matrix_semidefinite(puntos_media2, np.array([-3,3]),array = True , num_points = 100),axis=0)
        data = np.append(data,self.random_mean_and_matrix_semidefinite(puntos_media3, np.array([-1,1]),array = True , num_points = 100),axis=0)
        data = np.array(data)
        return data
    
    def contamination_generator(self,partition = 4, num_outliers_total =100):
        '''
    
        '''
        
        contamination_total = []
        num_outliers_part = num_outliers_total/partition
        for i in range (partition):
            contamination = []
            list_partition = [x for x in range(4)]
            contamination.append(np.random.uniform(low=-7,high=-2.5,size=num_outliers_part).reshape(num_outliers_part,1))
            contamination.append(np.random.uniform(low=-2,high=0,size=num_outliers_part).reshape(num_outliers_part,1))
            contamination.append(np.random.uniform(low=2,high=-3.7,size=num_outliers_part).reshape(num_outliers_part,1))
            contamination.append(np.random.uniform(low=5,high=7,size=num_outliers_part).reshape(num_outliers_part,1))
            contamination = np.array(contamination)
            shuffle(list_partition)
            contamination = np.concatenate((contamination[list_partition[0]],contamination[list_partition[1]],contamination[list_partition[2]],contamination[list_partition[3]]),axis=1)
            contamination_total.append(contamination)
        contamination_total = np.array(contamination_total)
        contamination = contamination_total[0]
        for i in range(1,contamination_total.shape[0]):
            contamination = np.append(contamination,contamination_total[i],axis=0)    
        
        return contamination
    


num_points = 100
num_outliers = 300

robust_generator = Robust()
#Se crea la contaminacion en ciertos puntos
data = robust_generator.data_generator()
contamination = robust_generator.contamination_generator(num_outliers_total = num_outliers)
np.savez('DataGaussian/dataExperiment2', mean = data[[0,2,4]] ,data = data[[1,3,5]][0], contamination = contamination)
#contaminacion = np.concatenate((contaminacion[0],contaminacion[1],contaminacion[2],contaminacion[3],contaminacion[4]),axis=1)
#np.concatenate((data[0],data[1],data[2]),axis=0).shape
