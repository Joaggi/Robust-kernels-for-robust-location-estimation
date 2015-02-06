import pylab as pl

def showHeat(K):
    fig = pl.figure(1)
    ax = fig.add_subplot(111)
    cax = ax.matshow(K, interpolation='nearest',vmax=50)
    pl.show()
