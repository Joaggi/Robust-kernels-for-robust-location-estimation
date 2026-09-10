# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:58:58 2014

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt

#huber
y = []
for j in range(1,4):
    x = np.linspace(-4,4,1000)
    K = j
    y.append([0.5*np.power(i,2) if np.abs(i) <= K else np.abs(i)*K - 0.5*np.power(K,2) for i in x])
    
ax = plt.subplot(1,2,1)
ax.plot(x, y[0], 'k--', label='K = 1')
ax.plot(x, y[1], 'k:', label='K = 2')
ax.plot(x, y[2], 'k', label='K = 3')  

ax.set_title("Huber contrast function")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho(x)$')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame  = legend.get_frame()
frame.set_facecolor('0.95')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('medium')

for label in legend.get_lines():
    label.set_linewidth(1)  # the legend line width
    
    
#Welsch
x = np.linspace(-4,4,1000)
y=1 - np.exp(-np.power(x,2)/2)
ax = plt.subplot(1,2,2)
ax.plot(x, y, 'k--')

ax.set_title("Welsch contrast function")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho(x)$')

plt.show()

    
    
#Andrews
a=1
y = []
for j in range(1,4):
    x = np.linspace(-4,4,1000)
    a = j
    y.append([a*(1-np.cos(i/a)) if np.abs(i) <= a*np.pi else 2*a for i in x])
    
ax = plt.subplot(2,2,2)
ax.plot(x, y[0], 'k--', label='a = 1')
ax.plot(x, y[1], 'k:', label='a = 2')
ax.plot(x, y[2], 'k', label='a = 3')  

ax.set_title("Andrews contrast function")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho(x)$')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame  = legend.get_frame()
frame.set_facecolor('0.95')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('medium')

for label in legend.get_lines():
    label.set_linewidth(1)  # the legend line width    

#Bisquare
y = []
for j in range(1,4):
    x = np.linspace(-4,4,1000)
    K = j
    y.append([1-np.power(1-np.power(i/K,2),3) if np.abs(i) <= K else 1 for i in x])
print len(y[0])
ax = plt.subplot(2,2,3)
ax.plot(x, y[0], 'k--', label='K = 1')
ax.plot(x, y[1], 'k:', label='K = 2')
ax.plot(x, y[2], 'k', label='K = 3')  

ax.set_title("Bisquare contrast function")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\rho(x)$')

# Now add the legend with some customizations.
legend = ax.legend(loc='lower left', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame  = legend.get_frame()
frame.set_facecolor('0.95')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('medium')

for label in legend.get_lines():
    label.set_linewidth(0.8)  # the legend line width
    
    
    



