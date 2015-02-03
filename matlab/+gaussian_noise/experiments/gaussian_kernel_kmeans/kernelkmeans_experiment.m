function correct = kernelkmeans_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

% addpath '/home/jagallegom/Algorithms/Matlab';

k = max(size(unique(labels)));
% X= L1norm(data);
X= data;


cont = 1
vect = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,0,1,2,3,4,5,6,7,8,9,10];
%vect = [-8]

purityVec = zeros(epocs*length(vect),1);
clusteringAccuracyVec = zeros(epocs*length(vect),1);
nmiVec = zeros(epocs*length(vect),1);
labels_total = zeros(epocs*21,length(labels));

purityMeanVec = zeros(length(vect),1);
clusteringAccuracyMeanVec = zeros(length(vect),1);
nmiMeanVec = zeros(length(vect),1);

puritySdVec = zeros(length(vect),1);
clusteringAccuracySdVec = zeros(length(vect),1);
nmiSdVec = zeros(length(vect),1);

try
    load(nameresults);
    cont = (l-1)*(epocs)+1;
catch
    l = 1
end

while(l<=length(vect))
    aux = cont
    for i=1:epocs
%         try
            K = kernelRBF(X',X',2^vect(l));
            size(K);
            pos = ceil(rand(1,k)*size(X',2));
            H = init(K,pos);
            k = length(unique(labels));
%           inx=kmeans(X,k,'Emptyaction','singleton'); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
            [H, labelsPred] = Kkmeans(H, K, epocs);
%             [ labelsPred,energy] = knkmeans(K, k);
            contRecur = 1
            while length(unique(labelsPred)) ~= k
%                 pos = ceil(rand(1,k)*size(X',2));
%                 H = init(K,pos);
%                 k = length(unique(labels));
    %             inx=kmeans(X,k,'Emptyaction','singleton'); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
                [H, labelsPred] = Kkmeans(H, K, epocs);
%                 [ labelsPred,energy] = knkmeans(K, k);
                contRecur  = contRecur +1
                if contRecur >epocs
                    break
                end
            end
%           H = init(K,pos);
%           [energy,labelsPred] = Kkmeans(H, K, 100);
%            clear energy;
	        labels_total(cont,:) = labelsPred;

            [result,confus] = ClusteringMeasure(labels,labelsPred)
            clusteringAccuracyVec(cont) = result(1)
            purityVec(cont) = result(3)
            nmiVec(cont) = result(2)
%         catch
%             sprintf('Ha ocurrido un error')
%             purityVec(cont) = 0;
%             nmiVec(cont) = 0;
%             clusteringAccuracyVec(cont) = 0;
%         end
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
