from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix

def accuracy(l,lg):
    profitMatrix = confusion_matrix(lg, l)
    costMatrix = lg.shape[0] - profitMatrix
    ind = linear_assignment(costMatrix)
    total = 0.0
    for i in ind:
        total += profitMatrix[i[0],i[1]]
    return total / lg.shape[0]
