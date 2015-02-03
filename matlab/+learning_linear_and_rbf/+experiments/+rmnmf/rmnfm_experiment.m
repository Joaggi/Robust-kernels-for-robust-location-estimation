function correct = rmnmf_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')


vectLambda = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3]
rho = [1.05,1.15,1.25,1.35,1.45,1.66,1.8,2.2,2.5,3]


% vectLambda = [10^-3]
% rho = [1.66]

[X,Y] = meshgrid(vectLambda,rho);
parameters = [X(:) Y(:)];

num = size(parameters,1);

addpath '../../../Algorithms/Matlab';

k = max(size(unique(labels)));
% X= L1norm(data);
X = data;
% X= standarization(data');
% X = X'


purityVec = zeros(epocs*num,1);
clusteringAccuracyVec = zeros(epocs*num,1);
nmiVec = zeros(epocs*num,1);
labels_total = zeros(epocs*num,length(labels));


purityMeanVec = zeros(num,1);
clusteringAccuracyMeanVec = zeros(num,1);
nmiMeanVec = zeros(num,1);

puritySdVec = zeros(num,1);
clusteringAccuracySdVec = zeros(num,1);
nmiSdVec = zeros(num,1);

cont = 1;


try
    load(nameresults)
    cont = (l-1)*(epocs)+1
catch
    l = 1
end

while(l<=num)
    aux=cont
    for i=1:epocs
            [labels_pred] = RMNMF(X',labels,k,10^-3,parameters(l,1), parameters(l,2));
	        labels_total(cont,:) = labels_pred;
            [result,~] = ClusteringMeasure(labels,labels_pred);
            clusteringAccuracyVec(cont) = result(1);
            purityVec(cont) = result(3);
            nmiVec(cont) = result(2);
            cont = cont + 1;
    end
    purityMeanVec(l) = mean(purityVec(aux:cont-1));
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
    nmiMeanVec(l) = mean(nmiVec(aux:cont-1));  
    save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','puritySdVec','clusteringAccuracySdVec','nmiSdVec','l')
end
save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
