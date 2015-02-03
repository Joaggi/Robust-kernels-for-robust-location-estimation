function correct = kernelconvexnmf_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')
% 
addpath '../../Algorithms/Matlab';
addpath '../../Algorithms/Matlab/nmfv1_4';

k = length(unique(labels));
%X= L1norm(data);
%X= normalizeByRange(data,1);

cont = 1
vect = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,0,1,2,3,4,5,6,7,8,9,10];
% vect = [-7]

purityVec = zeros(epocs*length(vect),1);
clusteringAccuracyVec = zeros(epocs*length(vect),1);
nmiVec = zeros(epocs*length(vect),1);
labels_total = zeros(epocs*length(vect),length(labels));

purityMeanVec = zeros(length(vect),1);
clusteringAccuracyMeanVec = zeros(length(vect),1);
nmiMeanVec = zeros(length(vect),1);

puritySdVec = zeros(length(vect),1);
clusteringAccuracySdVec = zeros(length(vect),1);
nmiSdVec = zeros(length(vect),1);

option.kernel = 'rbf'
option.iter=500;
option.dis=1;
option.residual=1e-6;
option.tof=1e-6;

try
    load(nameresults)
    cont = (l-1)*(epocs)+1
catch
    l = 1
end

   
while(l<=length(vect))
    aux = cont
    option.param = 2^vect(l);
    for i=1:epocs
            labels_pred = kernelconvexnmfCluster(X',k,option);
            aux2 = 1;
            while length(unique(labels_pred)) ~= k
                labels_pred = kernelconvexnmfCluster(X',k,option);
                aux2 = aux2+1
                if aux2 >epocs
                    break
                end
            end
            labels_total(cont,:) = labels_pred;
            [result,confus] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1)
            purityVec(cont) = result(3)
            nmiVec(cont) = result(2)
        cont = cont + 1
    end
    purityMeanVec(l) = mean(purityVec(aux:cont-1))
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1))
    nmiMeanVec(l) = mean(nmiVec(aux:cont-1))
    puritySdVec(l) = std(purityVec(aux:cont-1))
    clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1))
    nmiSdVec(l) = std(nmiVec(aux:cont-1))
    l=l+1
    save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','l','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
    
end
save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','l','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
