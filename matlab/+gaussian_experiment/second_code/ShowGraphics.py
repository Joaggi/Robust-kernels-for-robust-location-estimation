# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:42:59 2014

@author: Alejandro
"""

n_subplots()
fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
m = 'o'
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, marker=m)
ax.set_xlabel('First Dimension')
ax.set_ylabel('Second Dimension')
ax.set_zlabel('Third Dimension')
ax.set_title('Numbers of outliers: 0' )

ax = fig.add_subplot(1,2,2, projection='3d')
m = 'o'
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, marker=m)
ax.set_xlabel('First Dimension')
ax.set_ylabel('Second Dimension')
ax.set_zlabel('Third Dimension')
ax.set_title('Numbers of outliers: all' )

ax.scatter(X_contamination[:,0], X_contamination[:,1], X_contamination[:,2], \
    c=['g' for i in range(X_contamination[:,0].shape[0])], marker='^')
    
plt.show()