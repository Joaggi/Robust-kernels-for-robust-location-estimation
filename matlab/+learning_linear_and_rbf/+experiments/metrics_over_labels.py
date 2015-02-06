# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:50:05 2014

@author: Alejandro
"""

import scipy.io as sio
import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)
    
import numpy as np
import Algorithms.Python.Metrics as Metrics
    
def Kernelconvexnmf(path,labels_name):

    labels_pred_total = sio.loadmat(path + '/Kernelconvexnmf/results_aux')['labels_total']
    labels_true = sio.loadmat(path + labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(21*30)
    calculate_purity = np.zeros(21*30)
    calculate_nmi = np.zeros(21*30)
    
    clustering_accuracy_mean = np.zeros(21)
    calculate_purity_mean = np.zeros(21)
    calculate_nmi_mean = np.zeros(21)
    
    clustering_accuracy_sd = np.zeros(21)
    calculate_purity_sd = np.zeros(21)
    calculate_nmi_sd = np.zeros(21)
    
    cont = 0
    contador = 0
    for j in range(21):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path + 'Kernelconvexnmf/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})
    
def KernelKMeans(path,labels_name):
    labels_pred_total = sio.loadmat(path + 'KernelKMeans/results_aux')['labels_total']
    labels_true = sio.loadmat(path + labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(21*30)
    calculate_purity = np.zeros(21*30)
    calculate_nmi = np.zeros(21*30)
    
    clustering_accuracy_mean = np.zeros(21)
    calculate_purity_mean = np.zeros(21)
    calculate_nmi_mean = np.zeros(21)
    
    clustering_accuracy_sd = np.zeros(21)
    calculate_purity_sd = np.zeros(21)
    calculate_nmi_sd = np.zeros(21)
    
    cont = 0
    contador = 0
    for j in range(21):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path + 'KernelKMeans/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})

def Kernelseminmfnnls(path,labels_name):
    labels_pred_total = sio.loadmat(path+'Kernelseminmfnnls/results_aux')['labels_total']
    labels_true = sio.loadmat(path+labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(21*30)
    calculate_purity = np.zeros(21*30)
    calculate_nmi = np.zeros(21*30)
    
    clustering_accuracy_mean = np.zeros(21)
    calculate_purity_mean = np.zeros(21)
    calculate_nmi_mean = np.zeros(21)
    
    clustering_accuracy_sd = np.zeros(21)
    calculate_purity_sd = np.zeros(21)
    calculate_nmi_sd = np.zeros(21)
    
    cont = 0
    contador = 0
    for j in range(21):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            #if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
            clustering_accuracy[(j)*21+i] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
            calculate_purity[(j)*21+i],vector = Metrics.calculate_purity(labels_true,labels_pred)
            calculate_nmi[(j)*21+i] = Metrics.calculate_nmi(labels_true,labels_pred)
            cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[(j)*21:(j)*21+30])
        calculate_purity_mean[j] =  np.mean(calculate_purity[(j)*21:(j)*21+30])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[(j)*21:(j)*21+30])
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[(j)*21:(j)*21+30])
        calculate_purity_sd[j] =  np.std(calculate_purity[(j)*21:(j)*21+30])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[(j)*21:(j)*21+30])
        
    
    sio.savemat(path+'Kernelseminmfnnls/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})
def Kernelseminmfrule(path,labels_name):
    labels_pred_total = sio.loadmat(path+'Kernelseminmfrule/results_aux')['labels_total']
    labels_true = sio.loadmat(path+labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(21*30)
    calculate_purity = np.zeros(21*30)
    calculate_nmi = np.zeros(21*30)
    
    clustering_accuracy_mean = np.zeros(21)
    calculate_purity_mean = np.zeros(21)
    calculate_nmi_mean = np.zeros(21)
        
    clustering_accuracy_sd = np.zeros(21)
    calculate_purity_sd = np.zeros(21)
    calculate_nmi_sd = np.zeros(21)
    
    cont = 0
    contador = 0
    for j in range(21):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        

        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path+'Kernelseminmfrule/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})


def KMeans(path,labels_name):
    labels_pred_total = sio.loadmat(path+'KMeans/results_aux')['labels_total']
    labels_true = sio.loadmat(path+labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(1*30)
    calculate_purity = np.zeros(1*30)
    calculate_nmi = np.zeros(1*30)
    
    clustering_accuracy_mean = np.zeros(1)
    calculate_purity_mean = np.zeros(1)
    calculate_nmi_mean = np.zeros(1)
        
    clustering_accuracy_sd = np.zeros(1)
    calculate_purity_sd = np.zeros(1)
    calculate_nmi_sd = np.zeros(1)
    
    cont = 0
    contador = 0
    for j in range(1):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path+'KMeans/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})
def NNMF(path,labels_name):
    labels_pred_total = sio.loadmat(path+'NNMF/results_aux')['labels_total']
    labels_true = sio.loadmat(path+labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    
    clustering_accuracy = np.zeros(1*30)
    calculate_purity = np.zeros(1*30)
    calculate_nmi = np.zeros(1*30)
    
    clustering_accuracy_mean = np.zeros(1)
    calculate_purity_mean = np.zeros(1)
    calculate_nmi_mean = np.zeros(1)
        
    clustering_accuracy_sd = np.zeros(1)
    calculate_purity_sd = np.zeros(1)
    calculate_nmi_sd = np.zeros(1)
    
    cont = 0
    contador = 0
    for j in range(1):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path+'NNMF/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})

def RMNMF(path,labels_name):
    labels_pred_total = sio.loadmat(path+'RMNMF/results_aux')['labels_total']
    labels_true = sio.loadmat(path+labels_name)['labels']
    
    print labels_pred_total.shape
    print labels_true.shape
    num = 4
    epocs = 30
    
    clustering_accuracy = np.zeros(num*epocs)
    calculate_purity = np.zeros(num*epocs)
    calculate_nmi = np.zeros(num*epocs)
    
    clustering_accuracy_mean = np.zeros(num)
    calculate_purity_mean = np.zeros(num)
    calculate_nmi_mean = np.zeros(num)
        
    clustering_accuracy_sd = np.zeros(num)
    calculate_purity_sd = np.zeros(num)
    calculate_nmi_sd = np.zeros(num)
    
    cont = 0
    contador = 0
    for j in range(num):
        aux = cont
        for i in range(30):
            labels_pred = labels_pred_total[contador,:]
            if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                cont+=1
            contador+=1
        clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
        calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
        calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
        
        clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
        calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
        calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
    sio.savemat(path+'RMNMF/results',{'labels_total' : labels_pred_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})

def metrics(path,dictionary,labels_name):
    if dictionary['Kernelconvexnmf']:
        Kernelconvexnmf(path,labels_name)
    if dictionary['KernelKMeans']:    
        KernelKMeans(path,labels_name)
    if dictionary['Kernelseminmfnnls']:            
        Kernelseminmfnnls(path,labels_name)
    if dictionary['Kernelseminmfrule']:        
        Kernelseminmfrule(path,labels_name)
    if dictionary['KMeans']:        
        KMeans(path,labels_name)
    if dictionary['NNMF']:        
        NNMF(path,labels_name)
    if dictionary['RMNMF']:        
        RMNMF(path,labels_name)
    