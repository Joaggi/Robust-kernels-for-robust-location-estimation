function correct = kmeans_experiment(filedata,dataset,datalabels,fileresults)

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

epocs = 200

addpath '../../../Algorithms/Matlab';

k = length(unique(labels));
% X= data;
X= normalizeByRange(data,1);



porcentajeContaminacion= 30
%Definicion de variables
[numContamination,vectorCentroids,realCentroids,biasContamination,biasVecInitial,biasGlobal] =   definicionVariables(X,labels,k,epocs,porcentajeContaminacion);

cont = 1

biasAux = Inf;
for i=1:epocs    
        k = length(unique(labels));
        [labelsPred,C] = kmeansAlgorithm(X,k);
        vectorCentroids(:,:,i) = geneticAlgorithm(realCentroids,C);
        biasVecInitial(i) = sum(sum(abs(vectorCentroids(:,:,i) - realCentroids)));
        
        if biasVecInitial(i) < biasAux
            labelsReal = labelsPred;
            biasAux = biasVecInitial(i);
        end
    cont = cont + 1
end
centroid = mean(vectorCentroids,3);
centroidInitial = centroid;
minFeature = min(X)-min(X)*0.5;
maxFeature = max(X)+max(X)*0.5;
% vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));
    
for j= 1:porcentajeContaminacion
    vectorCentroids = zeros(k,size(X,2),epocs);
    XwithContamination = [X;vectorContamination(1:floor(size(X,1)*(j/100)),:)];
    for i=1:epocs       
            k = length(unique(labels))
            [~,C] = kmeansAlgorithm(XwithContamination,k);
            vectorCentroids(:,:,i) = geneticAlgorithm(centroidInitial,C);
    end
    centroid = mean(vectorCentroids,3);
    biasGlobal(j) = sum(sum(abs(centroid - realCentroids)));
    biasContamination(j) =  sum(sum(abs(centroid - centroidInitial)));
end


save(fileresults,'vectorCentroids','realCentroids','X','biasGlobal','vectorContamination','biasContamination')
end


function centroids = get_centroids(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels))
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end


function [numContamination,vectorCentroids,realCentroids,biasContamination,biasVecInitial,biasGlobal] =   definicionVariables(X,labels,k,epocs,porcentajeContaminacion)
numContamination = floor(porcentajeContaminacion/100*size(X,1));
vectorCentroids = zeros(k,size(X,2),epocs);
realCentroids = get_centroids(labels,X);
biasVecInitial = zeros(epocs,1)

biasGlobal = zeros(porcentajeContaminacion,1);
biasContamination = zeros(porcentajeContaminacion,1);
end

function [labelsPred,C] = kmeansAlgorithm(X,k)
        [labelsPred,C] =kmeans(X,k,'Emptyaction','singleton');
        contRecur = 0;
        while length(unique(labelsPred)) ~= k
            [labelsPred, C] = kmeans(X,k,'Emptyaction','singleton');
            clear energy;
            if contRecur >30
                break
            end
            contRecur  = contRecur +1
        end
end
