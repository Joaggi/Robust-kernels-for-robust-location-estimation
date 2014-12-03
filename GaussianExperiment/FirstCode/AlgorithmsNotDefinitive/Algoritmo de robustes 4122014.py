# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:03:49 2014

@author: Alejandro
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 18:58:38 2014

@author: Alejandro
"""
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from kkmeans import *
from sklearn.metrics import pairwise as pw

from scipy import random

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

def generar_numeros_aleatorios():
    
    matrixSize = 10 
    A = random.rand(matrixSize,matrixSize)
    B = A*A.transpose()
    print 'random positive semi-define matrix for today is', B


puntos_media = np.array([[-5,-4],[2,2.5],[1,2],[4,5]])
puntos_varianza = np.array([[-2,2],[1,1],[0.5,1],[1,0.5]])


mean1 = np.random.uniform(low=puntos_media[0,0],high=puntos_media[0,1],size=4)
cov1 = np.random.uniform(low=puntos_varianza[0,0],high=puntos_varianza[0,1],size=(4,4))
mean2 = np.random.uniform(low=puntos_media[1,0],high=puntos_media[1,1],size=4)
cov2 = np.random.uniform(low=puntos_varianza[1,0],high=puntos_varianza[1,1],size=(4,4))
mean3 = np.random.uniform(low=puntos_media[2,0],high=puntos_media[2,1],size=4)
cov3 = np.random.uniform(low=puntos_varianza[2,0],high=puntos_varianza[2,1],size=(4,4))
mean4 = np.random.uniform(low=puntos_media[3,0],high=puntos_media[3,1],size=4)
cov4 = np.random.uniform(low=puntos_varianza[3,0],high=puntos_varianza[3,1],size=(4,4))

numdatos = 100
numiter = 300
ncluster= 3
numiter_conv = 1000


sum_kmeans = []
sum_kernelkmeans=[]
sesgo_kernelkmeans= []
sesgo_kmeans = []

num_outliers_grafica= 40

for l in xrange(num_outliers_grafica):
    print l
    numoutliers = l*4
    sum_kmeans.append([0])
    sum_kernelkmeans.append([0])
    sesgo_kernelkmeans.append([0])
    sesgo_kmeans.append([0])
    for i in xrange(numiter):
        
        GM1 = np.random.multivariate_normal(mean1,cov1,numdatos)
        GM2 = np.random.multivariate_normal(mean2,cov2,numdatos)
        GM3 = np.random.multivariate_normal(mean3,cov3,numdatos)
        #plt.plot(x1,y1,'x')
        #plt.plot(x2,y2,'x')
        #plt.axis('equal')
        #plt.show()
        #contaminacion = np.concatenate((np.random.uniform(low=-8,high=8,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-8,high=8,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-8,high=8,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-8,high=8,size=numoutliers).reshape(numoutliers,1)),axis=1)
        contaminacion = np.concatenate((np.random.uniform(low=-5,high=-3,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-0,high=-2,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-3.5,high=-3.7,size=numoutliers).reshape(numoutliers,1),np.random.uniform(low=-4.78,high=5,size=numoutliers).reshape(numoutliers,1)),axis=1)
        
        X = np.concatenate((GM1,GM2,GM3,contaminacion),axis=0)
        X_without_contamination = np.concatenate((GM1,GM2,GM3),axis=0)
    
        
        k_means = KMeans(n_clusters = 3, max_iter = 10000, random_state = 0)
        k_means.fit_predict(X)
        k_means.predict(X)
        predict = k_means.predict(X_without_contamination)
        segundo = np.empty((1,numdatos))
        segundo[:] = 2
        vector1 = np.concatenate((np.zeros((1,numdatos)),np.ones((1,numdatos)),segundo),axis=1)
        vector2 = np.concatenate((np.zeros((1,numdatos)),segundo,np.ones((1,numdatos))),axis=1)
        vector3 = np.concatenate((np.ones((1,numdatos)),np.zeros((1,numdatos)),segundo),axis=1)
        vector4 = np.concatenate((np.ones((1,numdatos)),segundo,np.zeros((1,numdatos))),axis=1)
        vector5 = np.concatenate((segundo,np.zeros((1,numdatos)),np.ones((1,numdatos))),axis=1)
        vector6 = np.concatenate((segundo,np.ones((1,numdatos)),np.zeros((1,numdatos))),axis=1)
        #print(predict)
        error1 = np.sum(np.abs(predict - vector1))
        error2 = np.sum(np.abs(predict - vector2))
        error3 = np.sum(np.abs(predict - vector3))
        error4 = np.sum(np.abs(predict - vector4))
        error5 = np.sum(np.abs(predict - vector5))
        error6 = np.sum(np.abs(predict - vector6))
        #print np.sum(error)
        error = min(error1,error2,error3,error4,error5,error6)
        sum_kmeans[l] = np.sum(error) + sum_kmeans[l]
        error_2 = 10000
        for i in xrange(15):
            try:
                km = KernelKMeans(n_clusters=3, max_iter=1000, verbose=0,kernel='rbf',gamma=2**-i)
                km.fit_predict(X)
                km.predict(X)
                predict = km.predict(X_without_contamination)
                segundo = np.empty((1,numdatos))
                segundo[:] = 2
                vector1 = np.concatenate((np.zeros((1,numdatos)),np.ones((1,numdatos)),segundo),axis=1)
                vector2 = np.concatenate((np.zeros((1,numdatos)),segundo,np.ones((1,numdatos))),axis=1)
                vector3 = np.concatenate((np.ones((1,numdatos)),np.zeros((1,numdatos)),segundo),axis=1)
                vector4 = np.concatenate((np.ones((1,numdatos)),segundo,np.zeros((1,numdatos))),axis=1)
                vector5 = np.concatenate((segundo,np.zeros((1,numdatos)),np.ones((1,numdatos))),axis=1)
                vector6 = np.concatenate((segundo,np.ones((1,numdatos)),np.zeros((1,numdatos))),axis=1)
                #print(predict)
                error1 = np.sum(np.abs(predict - vector1))
                error2 = np.sum(np.abs(predict - vector2))
                error3 = np.sum(np.abs(predict - vector3))
                error4 = np.sum(np.abs(predict - vector4))
                error5 = np.sum(np.abs(predict - vector5))
                error6 = np.sum(np.abs(predict - vector6))
                #print np.sum(error)
                error = min(error1,error2,error3,error4,error5,error6)
                if error < error_2 : 
                    error_2 = error
                    kmmean = km
                    par_gamma = 2**-i
            except:
                #print "error"
                pass
            
            try:
                km = KernelKMeans(n_clusters=3, max_iter=1000, verbose=0,kernel='rbf',gamma=2**i)
                km.fit_predict(X)
                km.predict(X)
                predict = km.predict(X_without_contamination)
                segundo = np.empty((1,numdatos))
                segundo[:] = 2
                vector1 = np.concatenate((np.zeros((1,numdatos)),np.ones((1,numdatos)),segundo),axis=1)
                vector2 = np.concatenate((np.zeros((1,numdatos)),segundo,np.ones((1,numdatos))),axis=1)
                vector3 = np.concatenate((np.ones((1,numdatos)),np.zeros((1,numdatos)),segundo),axis=1)
                vector4 = np.concatenate((np.ones((1,numdatos)),segundo,np.zeros((1,numdatos))),axis=1)
                vector5 = np.concatenate((segundo,np.zeros((1,numdatos)),np.ones((1,numdatos))),axis=1)
                vector6 = np.concatenate((segundo,np.ones((1,numdatos)),np.zeros((1,numdatos))),axis=1)
                #print(predict)
                error1 = np.sum(np.abs(predict - vector1))
                error2 = np.sum(np.abs(predict - vector2))
                error3 = np.sum(np.abs(predict - vector3))
                error4 = np.sum(np.abs(predict - vector4))
                error5 = np.sum(np.abs(predict - vector5))
                error6 = np.sum(np.abs(predict - vector6))
                #print np.sum(error)
                error = min(error1,error2,error3,error4,error5,error6)
                if error < error_2 :
                    error_2 = error
                    kmmean = km
                    par_gamma = 2**-i
            except:
                #print "error"
                pass
        #print error_2
        sum_kernelkmeans[l] = np.sum(error_2) + sum_kernelkmeans[l]
        
        #Pre imagen del kernel k means
        z = np.random.uniform(low=-10,high=10,size=X.shape[1]).reshape((1,X.shape[1]))
        for i in xrange(ncluster-1):
            z = np.concatenate((z,np.random.uniform(low=-10,high=10,size=X.shape[1]).reshape((1,X.shape[1]))),axis=0)
        print "z inicial"
        print z
        predict = kmmean.predict(X)
        clusters_indices = []
        for i in xrange(ncluster):
            clusters_indices.append([])
        for j in xrange(ncluster):
            clusters_indices[j] = np.where(predict == j)
            #print clusters_indices
        
        for i in xrange(numiter_conv):
            for j in xrange(ncluster):
                kernel = rbf_kernel(z[j],X[clusters_indices[j]],par_gamma)         
                #print kernel.shape
                #print X[clusters_indices[j]].shape
                for k in xrange(X.shape[1]):
                    z[j,k] = np.dot(np.transpose(kernel),np.transpose(X[clusters_indices[j],k])) / np.sum(kernel)
                #print kernel
        print "Medias de z"
        print z
        print "Medias reales"
        print mean1
        print mean2
        print mean3
        
        a  = kmmean.predict(X[0,:])
        print "kmmean 1"
        print a
        a  = kmmean.predict(X[141,:])
        print "kmmean 2"
        print a
        
        for i in range(ncluster):
            sesgo_kmeans[l] = np.abs(np.min((np.sum(k_means.cluster_centers_[i]-mean1),np.sum(k_means.cluster_centers_[i]-mean2),np.sum(k_means.cluster_centers_[i]-mean3))))+sesgo_kmeans[l]
                
        
        for i in range(ncluster):
            sesgo_kernelkmeans[l] = np.abs(np.min((np.sum(z[i]-mean1),np.sum(z[i]-mean2),np.sum(z[i]-mean3))))+sesgo_kernelkmeans[l]
    sesgo_kmeans[l] = sesgo_kmeans[l] / numiter    
    sesgo_kernelkmeans[l] = sesgo_kernelkmeans[l]/numiter
    sum_kmeans[l] = sum_kmeans[l]/numiter
    sum_kernelkmeans[l] = sum_kernelkmeans[l]/numiter
print sum_kmeans
print sum_kernelkmeans


import pylab as pl
pl.clf()
pl.plot(range(num_outliers_grafica),sesgo_kmeans)
pl.plot(range(num_outliers_grafica),sesgo_kernelkmeans)


pl.clf()
pl.plot(range(num_outliers_grafica),sum_kmeans)
pl.plot(range(num_outliers_grafica),sum_kernelkmeans)