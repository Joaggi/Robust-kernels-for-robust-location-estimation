__authors__ = "Joseph Gallego"
__email__ = "jagallegom@unal.edu.co"

import scipy.io as sio
import semantic_methods_toolkit.python.dataset_factory as dataset_factory
from semantic_methods_toolkit.python.cnmf_experiment import CnmfExperiment
from semantic_methods_toolkit.python.contamination_experiment import contamination_experiment
from semantic_methods_toolkit.python.generate_logarithm_vector_kernel import GenerateLogarithmVectorKernel
from semantic_methods_toolkit.python.preprocess_data.normalize_by_range import NormalizeByRange


data,labels = dataset_factory.dataset_factory('../../dataset/jaffe_occlusion_contamination.mat',options = {'data': 'data_real','labels':'labels'})
data_contamination,labels = dataset_factory.dataset_factory('../../dataset/jaffe_occlusion_contamination.mat',options = {'data': 'data_contamination','labels':'labels'})

import numpy as np

print data
k = np.unique(labels).size

data = NormalizeByRange(data,axis=0)   

vect = GenerateLogarithmVectorKernel(data)

dt = [('key', 'S30'), ('value', 'S30')]
arr = np.zeros((10,), dtype=dt)

arr[0]['value'] = str(k)
arr[0]['key'] = 'k'
arr[1]['value'] = "iter: " + str(1000)
arr[1]['key'] = 'termination_criterion'
arr[2]['value'] = 'random'
arr[2]['key'] = 'initialization'
arr[3]['value'] = '1'
arr[3]['key'] = 'epocs'
arr[4]['value'] = '1'
arr[4]['key'] = 'clustering_accuracy'
arr[5]['value'] = '1'
arr[5]['key'] = 'purity'
arr[6]['value'] = 'rbf'
arr[6]['key'] = 'kernel'
arr[7]['value'] = '1'
arr[7]['key'] = 'param'
arr[8]['value'] = str([-5,-4,-3,-2,-1,0,1,2,3,4,5])
arr[8]['key'] = 'vect'
arr[8]['value'] = '20'
arr[8]['key'] = 'percentage_contamination'

def options(vect):
    option = {arr[0]['key'] : int(arr[0]['value']),
        arr[1]['key'] : {'iter':1000},
        arr[2]['key'] : arr[2]['value'],
        arr[3]['key'] : int(arr[3]['value']),
        arr[4]['key'] : int(arr[4]['value']),
        arr[5]['key'] : int(arr[5]['value']),
        arr[6]['key'] : arr[6]['value'],
        arr[7]['key'] : int(arr[7]['value']),
        arr[8]['key'] : int(arr[8]['value'])
        }
    for i in vect:
        #define sigma
        option['vect'] = vect
        yield option
        

option = {
        arr[0]['key'] : int(arr[0]['value']),
        arr[1]['key'] : {'iter':1000},
        arr[2]['key'] : arr[2]['value'],
        arr[3]['key'] : int(arr[3]['value']),
        arr[4]['key'] : int(arr[4]['value']),
        arr[5]['key'] : int(arr[5]['value']),
        arr[6]['key'] : arr[6]['value'],
        arr[7]['key'] : int(arr[7]['value']),
        arr[8]['key'] : int(arr[8]['value'])
        }
        

cnmf_experiment = CnmfExperiment(data,option)
      
results_performance, last_results_performance , options_results_performance,   results_performance_tunning = contamination_experiment(
    data,data_contamination, labels, cnmf_experiment, options([1,2,3]) , option)



import time
## dd/mm/yyyy format
date =  (time.strftime("%d%m%Y_%H:%M_"))

np.save( str(date) +'linear_cnmf' , {'last_results_performance':last_results_performance,
                    'results_performance':results_performance, 'results_performance_tunning': results_performance_tunning,
                    'options_results_performance': options_results_performance,'options':arr})
