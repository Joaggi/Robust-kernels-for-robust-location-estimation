# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""


import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)
import Robustes.Experiments.KernelKMeans.kernelkmeans_experiment as kernelkmeans

filedata = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Wineq/winequality-white.npz'
epocs  = 'G:/Dropbox/Universidad/Machine Learning/Robustes/Wineq/parameters.mat'

kernelkmeans.kernelkmeans(filedata,epocs,normalized_axis = 0 ,norm='l1')