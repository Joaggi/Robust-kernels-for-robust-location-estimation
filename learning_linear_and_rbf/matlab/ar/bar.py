# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:31:02 2014

@author: Alejandro
"""

#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)   

import numpy as np
import Robustes.Experiments.bar as bar
    

bar.bar_accuracy('G:/Dropbox/Universidad/Machine Learning/Robustes/AR/','AR')