import itertools

import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl

from sklearn import mixture

def generateGMM(samples,features,separation,means,plotit):
    C = np.array([[1., 0], [0, 1.1]])
    X = np.r_[np.dot(np.random.randn(samples, features) + np.multiply(separation,4*np.array(np.ones(features))), C),
          1.4 * np.random.randn(samples, features) + np.array(means)]
    Y = np.r_[np.dot(np.random.randn(samples, features), C),
          1.4 * np.random.randn(samples, features) + np.array(means)]
    if plotit == True:
        gmm = mixture.GMM(n_components=2, covariance_type='full')
        gmm.fit(X)

        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

        for i, (clf, title) in enumerate([(gmm, 'GMM')]):
            splot = pl.subplot(2, 1, 1 + i)
            Y_ = clf.predict(X)
            for i, (mean, covar, color) in enumerate(zip(
                    clf.means_, clf._get_covars(), color_iter)):
                v, w = linalg.eigh(covar)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                if not np.any(Y_ == i):
                    continue
                pl.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)

            pl.xlim(-5, 5)
            pl.ylim(-2, 18)
            pl.xticks(())
            pl.yticks(())
            pl.title(title)

        pl.show()
        return [False,False]
    else:
        return [X,Y]
