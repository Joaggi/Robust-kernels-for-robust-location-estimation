"""Kernel K-means"""
 
# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause
 
import numpy as np
 
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.utils import as_float_array

def trimmedrbf_kernel(X, Y=None, gamma=None, robust_gamma = None):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||**2)

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    print K
    print "SHape kernel" + str(np.where(np.sqrt(K) > robust_gamma)[0].shape)
    K[np.where(np.sqrt(K) > robust_gamma)] = robust_gamma**2
    
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    return K

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """
 
    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0 , robust_gamma = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.robust_gamma = robust_gamma
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"
 
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        if self.kernel == "robust_kernel":
            return trimmedrbf_kernel(X, Y, gamma=self.gamma, robust_gamma = self.robust_gamma)                  
        else:
            return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)
 
    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]
 
        K = self._get_kernel(X)
 
        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw
 
        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)
 
        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)
 
        for it in xrange(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)
 
            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
#            if 1 - float(n_same) / n_samples < self.tol:
#                if self.verbose:
#                    print "Converged at iteration", it + 1
#                break
 
        self.X_fit_ = X
 
        return self
 
    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""
        sw = self.sample_weight_
 
        for j in xrange(self.n_clusters):
            mask = self.labels_ == j
 
            if np.sum(mask) == 0:
                dist[:, j] = np.finfo('d').max
                #continue
                raise ValueError("Empty cluster found, try smaller n_cluster.")
 
            denom = sw[mask].sum()
            denomsq = denom * denom
 
            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]
 
            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom
        #print "\n distancia \n"
        #print dist
 
    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        #print " Kernel \n"
        #print K
        #print K.shape
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)
 
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=5, random_state=0)
 
    km = KernelKMeans(n_clusters=5, max_iter=100, random_state=0, verbose=1)
    print km.fit_predict(X)[:10]
    print km.predict(X[:10])
    
    

def get_kernelKMeans(fileData,normalized_axis=1,norm='l1',gamma = None):

    data = np.load(fileData)
    #print data
    features = data['data']
    n_clusters = np.size(np.unique(data['labels']))
    features = as_float_array(features)
    print features
    print features.shape
    if normalized_axis != None:
        features = normalize(features,norm=norm,axis=normalized_axis)
    print features
    model = KernelKMeans(n_clusters=n_clusters, 
                             kernel="rbf", gamma=gamma,
                             verbose=1)
    print model
    model.fit(features)
    labels_pred = model.labels_
    labels_true = data['labels']
    return labels_true, labels_pred,features
    
def robust_kernel(fileData,normalized_axis=1,norm='l1',gamma = None , robust_gamma = 0.5):

    data = np.load(fileData)
    #print data
    features = data['data']
    n_clusters = np.size(np.unique(data['labels']))
    features = as_float_array(features)
    print features
    print features.shape
    if normalized_axis != None:
        features = normalize(features,norm=norm,axis=normalized_axis)
    print features
    model = KernelKMeans(n_clusters=n_clusters, 
                             kernel="robust_kernel", gamma=gamma,
                             verbose=1 , tol= 1e-4,robust_gamma = robust_gamma)
    print model
    model.fit(features)
    labels_pred = model.labels_
    labels_true = data['labels']
    return labels_true, labels_pred,features
    
    
