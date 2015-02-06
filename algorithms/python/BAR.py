# -*- coding: utf-8 -*-
"""
Created on Fri Jul 04 18:12:52 2014

@author: Alejandro
"""

accuracy = [0.546,0.5329,0.5236,0.5317]
nmi = [0.1652,0.1650,0.1483,0.1649] 
purity = [0.6055,0.5978,0.7261,0.5880]
import matplotlib.pylab as plt

fig = plt.figure('Abalone')
fig.text(.5, .95, 'Abalone') 
ax = fig.add_subplot(1,3,1)

ind = np.arange(4)
width = 0.8
ax.bar(ind,accuracy,color = ['b','g','r','y'])

ax.set_title('Cluster Accuracy')
plt.xticks(ind+width/2., ('KMeans', 'KernelKMeans', 'NMF', 'RNMF') )

ax = fig.add_subplot(1,3,2)
ax.bar(ind,nmi,color = ['b','g','r','y'])

ax.set_title('Norm Mutual Inf')
plt.xticks(ind+width/2., ('KMeans', 'KernelKMeans', 'NMF', 'RNMF') )

ax = fig.add_subplot(1,3,3)
ax.bar(ind,purity,color = ['b','g','r','y'])

ax.set_title('Purity')
plt.xticks(ind+width/2., ('KMeans', 'KernelKMeans', 'NMF', 'RNMF') )



plt.show()