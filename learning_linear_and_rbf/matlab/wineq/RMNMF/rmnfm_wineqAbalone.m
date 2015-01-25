addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
addpath 'G:/Dropbox/Universidad/Machine Learning/Robustes/Wineq/';
load('winequality-white')
load('winequality-white-labels')

k = max(size(unique(labels)));

X= L1norm(data);
%X = data;

[F,G] = variablesL21NMF(X,k);
G = G';
X = X';
fprintf('Legue')
[labels_pred] = L21NMF(X,F,G,0.01);
labels_pred = labels_pred'
purityvec = purity(labels,labels_pred)
clusteringAccuracy = clusteringaccuracy(labels,labels_pred)
nmivec = nmi(labels,labels_pred)
