# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""

import os, sys
lib_path = os.path.abspath('/home/jagallegom/')
sys.path.append(lib_path)
import Robustes.Experiments.KMeans.kmeans_experiment as kmeans

filedata = '/home/jagallegom/Robustes/Wineq/winequality-white.npz'
epocs  = '/home/jagallegom/Robustes/Wineq/parameters.mat'

kmeans.kmeans(filedata,epocs,100,normalized_axis = 0 ,norm='l1')