from hdf5 import *
import histersection as hi

def loadH5Dataset(h5datasetPath):
    """Loads a hdf5 file stored in h5datasetPath and returns T, Tp, V, and Vp matrices where its rows are samples"""
    h5_file = h5py.File(h5datasetPath, 'r')
    dataset = h5_file['/']
    T  = np.matrix(dataset['/textualTrain'])
    Tp = np.matrix(dataset['/textualTest'])
    V  = np.matrix(dataset['/visualTrain']) 
    Vp = np.matrix(dataset['/visualTest']) 
    return [T,Tp,V,Vp]

def buildKernels(T, Tp, V, Vp, types, normalizations):
    """Builds Kernels from features matrices T,Tp, V and Vp where
       their rows are samples.
       Inputs:                          
              T: Text matrix(lxn where l is the # of docs and n thr # of text features/labels), could be a binary matrix representing presence of label
              Tp: Test text matrix
              V: Visual Features matrix
              Vp: Test Visual Features matrix
              types: [typeForTextKernel, typeForVisualKernel] a list with the type for the text kernel and visual kernel
              normalizations: [normForTextFeatures, normForVisualFeatures, normForTextKernel, normForVisualKernel]

      Outputs: 
              Kt: Text training Kernel  of size lxl
              Kv: Visual Training Kernel of size lxl
              Kqv = Visual test  Kernel of size qxl
              Kqt = Textual test kernel of size qxl
"""
    if types[0]=="Linear": 
        Ktext  = T*np.transpose(T)
        

    if types[0]=="Histersection":
        Ktext  = hi.Histersection(T.T, T.T)

    if types[1]=="Linear": 
        Kvisual  = V*V.T
    if types[1]=="Histersection":
        Kvisual = hi.Histersection(V.T,V.T)
        
    if normalization =="L2":
        Ktn = np.divide(TT, np.sqrt(np.dot(TT.diagonal().transpose(),TT.diagonal()) )) #Normalization
    else:
        Ktn = np.zeros(10)
            
    return TT
