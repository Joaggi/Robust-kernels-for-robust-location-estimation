# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 23:56:43 2014

@author: joag
"""

import theano as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = function([x,y],z)
