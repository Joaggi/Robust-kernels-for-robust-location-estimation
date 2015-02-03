function correct = kernelkmeans_experiment(filedata,dataset,datalabels)

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

k = max(size(unique(labels)));
X= L1norm(data);

purityvec = zeros(epocs*20,1);
clusteringAccuracy = zeros(epocs*20,1);
nmivec = zeros(epocs*20,1);
cont = 1''
vect = [1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
for l = 1:20
    for i=1:epocs
        try
            K = gKernel(X,X,2^vect(l))
            [label_pred, energy]  = knkmeans(K,unique(labels))
            label_pred
            clear energy''
            purityvec(cont) = purity(labels,labels_pred);
            clusteringAccuracy(cont) = clusteringaccuracy(labels,labels_pred);
            nmivec(cont) = nmi(labels,labels_pred);
        catch
            purityvec(cont) = 0;
            nmivec(cont) = 0;
            clusteringAccuracy(cont) = 0;

        end
        cont = cont + 1
    end
    
end
save resultsKernelMatlab clusteringAccuracy purityvec nmivec
end
