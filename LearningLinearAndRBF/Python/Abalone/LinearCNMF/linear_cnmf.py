file_dataset = '../../../../datasets/abalone.mat'

import scipy as sio
import semantic_methods_toolkit.python.dataset_factory as dataset_factory
from semantic_methods_toolkit.python.cnmf_experiment import CnmfExperiment
from semantic_methods_toolkit.python.tunning_parameter_unsupervised_technique import tunning_parameter_unsupervised_technique

data,labels = dataset_factory.dataset_factory(file_dataset,options = {'data': 'data','labels':'labels'})
import numpy as np

print data
k = np.unique(labels).size

dt = [('key', 'S10'), ('value', 'S10')]
arr = np.zeros((5,), dtype=dt)

arr[0]['value'] = str(k)
arr[0]['key'] = 'k'
arr[1]['value'] = str(1000)
arr[1]['key'] = 'termination_criterion'
arr[2]['value'] = 'random'
arr[2]['key'] = 'initialization'
arr[3]['value'] = '30'
arr[3]['key'] = 'epocs'
arr[4]['value'] = '1'
arr[4]['key'] = 'clustering_accuracy'
arr[5]['value'] = '1'
arr[5]['key'] = 'purity'

options = {
        arr[0]['key'] : int(arr[0]['value']),
        arr[1]['key'] : {'iter':int(arr[1]['value'])},
        arr[2]['key'] : arr[2]['value']
        }
        
cnmf_experiment = CnmfExperiment(data,options)
      
def options(vect):
    option = {arr[0]['key']  : int(arr[0]['value']),
        arr[1]['key'] : {'iter':int(arr[1]['value'])},
        arr[2]['key']: arr[2]['value'],
        arr[3]['key']  : int(arr[3]['value']) ,
        arr[4]['key']  : int(arr[4]['value']),
        arr[5]['key'] : int(arr[5]['value'])
        }
    for i in vect:
        yield option

options_results_performance , results_performance = tunning_parameter_unsupervised_technique(data, labels, cnmf_experiment, options([1]))

import time
## dd/mm/yyyy format
date =  (time.strftime("%d/%m/%Y_%H:%M_"))

sio.savemat(str(date) + 'linear_cnmf' , {'options_results_performance':options_results_performance,
                    'results_performance':results_performance,'options':arr})
