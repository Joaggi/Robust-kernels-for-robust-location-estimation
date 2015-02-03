# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""

import os, sys
lib_path = os.path.abspath('/home/jagallegom/')
sys.path.append(lib_path)
import Robustes.Experiments.NMF.nmf_experiment as nmf

filedata = '/home/jagallegom/Robustes/AR/AR.npz'
epocs  = '/home/jagallegom/Robustes/AR/parameters.mat'

nmf.nmf(filedata,epocs,100,normalized_axis = 0 ,norm='l1')