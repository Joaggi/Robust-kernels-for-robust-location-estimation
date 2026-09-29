from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import check_pairwise_arrays

import numpy as np

def cauchy_kernel(X, Y=None, c=None):
    """
    Compute the cauchy kernel between X and Y::

        K(x, y) = -(c**2)/2 * log(1 + (||x-y||/c)**2)

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

    if c is None:
        c = X.mean() * 5
    K = euclidean_distances(X, Y, squared=False)

    gramMatrix = -np.power(c,2)/2.0 * np.log(1 + np.power(K/c,2))
    return gramMatrix
