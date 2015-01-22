# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""

import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)
import Robustes.Experiments.NMF.nmf_experiment as nmf

filedata = 'G:/Dropbox/Universidad/Machine Learning/Robustes/MovementLibras/movement_libras.npz'
epocs  = 'G:/Dropbox/Universidad/Machine Learning/Robustes/MovementLibras/parameters.mat'

nmf.nmf(filedata,epocs,normalized_axis = 0 ,norm='l1')