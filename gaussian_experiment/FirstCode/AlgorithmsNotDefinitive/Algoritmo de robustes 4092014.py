# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 18:58:38 2014

@author: Alejandro
"""
import numpy as np
from sklearn.cluster import KMeans
from kkmeans import *

puntos_media = np.array([[-5,-4],[-2,-1],[1,2],[4,5]])
puntos_varianza = np.array([[-2,2],[1,1],[0.5,1],[1,0.5]])


mean1 = np.random.uniform(low=puntos_media[0,0],high=puntos_media[0,1],size=4)
cov1 = np.random.uniform(low=puntos_varianza[0,0],high=puntos_varianza[0,1],size=(4,4))
mean2 = np.random.uniform(low=puntos_media[1,0],high=puntos_media[1,1],size=4)
cov2 = np.random.uniform(low=puntos_varianza[1,0],high=puntos_varianza[1,1],size=(4,4))
mean3 = np.random.uniform(low=puntos_media[2,0],high=puntos_media[2,1],size=4)
cov3 = np.random.uniform(low=puntos_varianza[2,0],high=puntos_varianza[2,1],size=(4,4))
mean4 = np.random.uniform(low=puntos_media[3,0],high=puntos_media[3,1],size=4)
cov4 = np.random.uniform(low=puntos_varianza[3,0],high=puntos_varianza[3,1],size=(4,4))

numdatos = 70
numoutliers = 100
numiter = 100


suma = 0
suma2=0
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

    
    k = KMeans(n_clusters = 3, max_iter = 10000, random_state = 0)
    k.fit_predict(X)
    k.predict(X)
    predict = k.predict(X_without_contamination)
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
    suma = np.sum(error) + suma
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
        except:
            print "error"
        
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
        except:
            print "error"
    #print error_2
    suma2 = np.sum(error_2) + suma2
    
    
    
suma = suma/numiter
suma2 = suma2/numiter
print suma
print suma2

a  = kmmean.predict(X[0,:])
print "kmmean 1"
print a
a  = kmmean.predict(X[141,:])
print "kmmean 2"
print a
