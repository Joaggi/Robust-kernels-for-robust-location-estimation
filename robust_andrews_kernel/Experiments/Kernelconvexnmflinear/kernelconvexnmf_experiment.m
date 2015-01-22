function correct = kernelconvexnmf_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';

% addpath '/home/jagallegom/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';

k = length(unique(labels));
%X= L1norm(data);
% X = data
X= normalizeByRange(data,1);


option.kernel = 'linear'
option.iter=3000;
vect = [0];


purityVec = zeros(epocs*length(vect),1);
clusteringAccuracyVec = zeros(epocs*length(vect),1);
nmiVec = zeros(epocs*length(vect),1);
labels_total = zeros(epocs*length(vect),length(labels));
finalResidualVec = zeros(epocs*length(vect),1);

purityMeanVec = zeros(length(vect),1);
clusteringAccuracyMeanVec = zeros(length(vect),1);
nmiMeanVec = zeros(length(vect),1);

puritySdVec = zeros(length(vect),1);
clusteringAccuracySdVec = zeros(length(vect),1);
nmiSdVec = zeros(length(vect),1);

cont = 1

try
    load(nameresults)
    cont = (l-1)*(epocs)+1
catch
    l = 1
end


while(l<=length(vect))
    aux = cont
    %option.param = 2^vect(l);
    i=1
    while i<=epocs
%          try
            %addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';    
            addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';
            [labels_pred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = kernelconvexnmfCluster(X',k,option);
            labels_total(cont,:) = labels_pred;
            %addpath '/home/jagallegom/Algorithms/Matlab';
            addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
            finalResidualVec(cont) = finalResidual
            [result,confus] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1)
            purityVec(cont) = result(3)
            nmiVec(cont) = result(2)
            i = i+1
            cont = cont + 1
%          catch
%             sprintf('Ha ocurrido un error')
%             purityVec(cont) = 0;
%             nmiVec(cont) = 0;
%             clusteringAccuracyVec(cont) = 0;            
%          end
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
