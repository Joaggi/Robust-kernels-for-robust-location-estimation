function correct = kernelkmeans_experiment(filedata,dataset,datalabels,nameresults)

%Carga
addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

epocs=50

addpath '../../../Algorithms/Matlab';

k = max(size(unique(labels)));
% X= data;
X= normalizeByRange(data,1);

vect = [-1,-2,-3,-4,-5,-6,0,1,2,3,4,5,6];
% vect = [5]

porcentajeContaminacion= 30
%Definicion de variables
[numContamination,vectorCentroids,realCentroids,centroids,vectorCentroidsContaminationTotal,bias,biasContamination] =   definicionVariables(X,vect,labels,k,epocs,porcentajeContaminacion);

cont = 1;
biasAux = inf;

numIterationsKKmeans = 300
numIterationsPreImage = 25

try
    load(nameresults);
    name = 1;
catch
    name = 0;
    j=1;
end

if name==0
    for l = 1:length(vect)
        aux = cont
        for i=1:epocs
            [H, labelsPred] = KernelKMeans(X,vect,l,k,numIterationsKKmeans);
            vectorCentroids(:,:,cont) = (preW(X', H, 2^vect(l),numIterationsPreImage))';
            vectorCentroids(:,:,cont) = geneticAlgorithm(realCentroids,vectorCentroids(:,:,cont));
            cont = cont + 1
        end
        centroids(:,:,l) = mean(vectorCentroids(:,:,(l-1)*epocs+1:(l)*epocs),3);
        bias(l) = sum(sum(abs(centroids(:,:,l) - realCentroids)));
        if bias(l) < biasAux
            labelsReal = labelsPred;
            biasAux = bias(l);
        end
    end
    [s,centroid,centroidInitial,biasInitial] = getResults(bias,centroids,realCentroids,vect);
    save(nameresults,'centroid','centroidInitial','biasInitial','realCentroids','centroids','bias','s','labelsReal','biasInitial','vectorCentroidsContaminationTotal','option');
end

minFeature = min(X)-min(X)*0.5;
maxFeature = max(X)+max(X)*0.5;

% vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));

cont = 1
vectorCentroidsContamination = zeros(k,size(X,2),epocs);

while(j<=porcentajeContaminacion)
    booleanVectorCentroids = ones(1,epocs);
    vectorCentroidsContamination = zeros(k,size(X,2),epocs);
    XwithContamination = [X;vectorContamination(1:floor(size(X,1)*(j/100)),:)];
        for i=1:epocs      
                [H, ~] = KernelKMeans(XwithContamination,vect,s,k,numIterationsKKmeans);
                vectorCentroidsContamination(:,:,i) = (preW(XwithContamination', H, 2^s,numIterationsPreImage))';
                vectorCentroidsContamination(:,:,i) = geneticAlgorithm(centroidInitial,vectorCentroidsContamination(:,:,i));
                cont = cont + 1
        end
        centroidContamination = mean(vectorCentroidsContamination(:,:,booleanVectorCentroids==1),3);
        biasContamination(j) =  sum(sum(abs(centroidContamination - centroidInitial)));
        biasGlobal(j) = sum(sum(abs(centroidContamination - realCentroids)));
        j=j+1
        save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','vectorContamination','biasContamination','biasGlobal','vectorCentroidsContamination','j','centroidInitial','s','labelsReal','option')
end

save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','vectorContamination','biasContamination','biasGlobal','vectorCentroidsContamination','j','centroidInitial','s','labelsReal','numIterationsKKmeans','numIterationsPreImage','option')
end

function centroids = get_centroids(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels));
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end

function [H, labelsPred] = KernelKMeans(X,vect,l,k,iter)
    K = gKernel(X',X',2^vect(l));
    pos = ceil(rand(1,k)*size(X,1));
    H = init(K,pos);    
    [H, labelsPred] = Kkmeans(H, K, iter);
    contRecur = 0
    while length(unique(labelsPred)) ~= k
        pos = ceil(rand(1,k)*size(X,1));
        H = init(K,pos);    
        [H, labelsPred] = Kkmeans(H, K, iter);
        if contRecur >30
            break
        end
        contRecur  = contRecur +1;
    end
end

function [numContamination,vectorCentroids,realCentroids,centroids,vectorCentroidsContaminationTotal,bias,biasContamination] =   definicionVariables(X,vect,labels,k,epocs,porcentajeContaminacion)
    numContamination = floor(porcentajeContaminacion/100*size(X,1));
    vectorCentroids = zeros(k,size(X,2),epocs*length(vect));
    realCentroids = get_centroids(labels,X);
    centroids  =  zeros(k,size(X,2),length(vect));
    vectorCentroidsContaminationTotal = zeros(k,size(X,2),porcentajeContaminacion);
    bias = zeros(length(vect),1);
    biasContamination = zeros(porcentajeContaminacion,1);
end

function [s,centroid,centroidInitial,biasInitial] = getResults(bias,centroids,realCentroids,vect)
    [~, posMin] = min(bias);
    s = vect(posMin);
    centroid = centroids(:,:,posMin);
    centroidInitial = centroid;
    biasInitial = sum(sum(abs(centroid - realCentroids)));
end
