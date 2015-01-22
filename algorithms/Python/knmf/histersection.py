import numpy as np
def histersection(A,B):
  K = []
  for v in A.T:
    K.append( 0.5*np.sum( v.A + B.T.A - np.abs(v.A - B.T.A), axis=1 ) )
  return np.asmatrix(K).T
