function correct = nmf_experiment(filedata,dataset,datalabels,nameresults)

addpath '../../../Algorithms/Matlab';
addpath '../../../Algorithms/Matlab/nmfv1_4';
addpath '../../../Datasets';

if(nargin < 4)
    nameresults = 'results';
end


addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

epocs = 100;

k = length(unique(labels));
%X= L1norm(data);
% X = data;
X= normalizeByRange(data,1);



porcentajeContaminacion= 30;
%Definicion de variables
[numContamination,vectorCentroids,realCentroids,vectorCentroidsContamination,bias,biasContamination,biasVecInitial,biasGlobal] =   definicionVariables(X,labels,k,epocs,porcentajeContaminacion);

cont = 1

option.reorder=false;
option.algorithm='nmfrule';
option.iterations=300;
option.iter=300;
option.distance = 'ls';
option.dis = false;

biasAux = Inf;
    for i=1:epocs
            k = length(unique(labels));
            [labelsPred,~,~,Yout,~,~,~] = nmfAlgorithm(X,k,option);
            vectorCentroids(:,:,i) = get_centroids(labelsPred,X,Yout);
            vectorCentroids(:,:,i) = geneticAlgorithm(realCentroids,vectorCentroids(:,:,i));
            biasVecInitial(i) = sum(sum(abs(vectorCentroids(:,:,i) - realCentroids)));
            
            if biasVecInitial(i) < biasAux
%                 labelsReal = labelsPred;
                biasAux = biasVecInitial(i);
            end
            cont = cont + 1
    end

centroid = mean(vectorCentroids,3);
centroidInitial = centroid;

biasInitial = sum(sum(abs(centroid - realCentroids)));

minFeature = min(X)-min(X)*0.5;
maxFeature = max(X)+max(X)*0.5;

% vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));
     

for j= 1:porcentajeContaminacion
    j
    vectorCentroidsContamination = zeros(k,size(X,2),epocs);
    booleanVectorCentroids = ones(1,epocs);
    XwithContamination = [X;vectorContamination(1:floor(size(X,1)*(j/100)),:)];
        for i=1:epocs      
            try
                k = length(unique(labels));
                [labelsPred,~,~,Yout,~,~,~] = nmfAlgorithm(XwithContamination,k,option);
                %[labelsPred,Yout] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),Yout);
                vectorCentroidsContamination(:,:,i) = get_centroids(labelsPred,XwithContamination,Yout);
                vectorCentroidsContamination(:,:,i) = geneticAlgorithm(centroidInitial,vectorCentroidsContamination(:,:,i));
                %[labelsPred,vectorCentroidsContamination(:,:,i)] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),vectorCentroidsContamination(:,:,i));
            catch
                sprintf('Ha ocurrido un error.');
                booleanVectorCentroids(i) = 0;
            end
        end
        centroidContamination = mean(vectorCentroidsContamination(:,:,booleanVectorCentroids==1),3);
        biasGlobal(j) = sum(sum(abs(centroid - realCentroids)));
        bias(j) = sum(sum(abs(centroid - centroidInitial)));
        biasContamination(j) =  sum(sum(abs(centroidContamination - centroidInitial)));

end


save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','biasGlobal','vectorContamination','biasContamination','vectorCentroidsContamination','epocs','option')
end


function centroids = get_centroids(labels,data,H)
    unique_labels = unique(labels);
    k = length(unique(labels));
    centroids = zeros(k,size(data,2));
   % H=H ./ repmat(sum(H,1),k,1)
    for i=1:k
        centroids(i,:)= sum(repmat(H(i,labels == unique_labels(i))',1,size(data,2)).*data(labels == unique_labels(i),:))/sum(H(i,labels == unique_labels(i)));
    end
end

function centroids = get_centroids_real(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels));
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end




function [numContamination,vectorCentroids,realCentroids,vectorCentroidsContamination,bias,biasContamination,biasVecInitial,biasGlobal] =   definicionVariables(X,labels,k,epocs,porcentajeContaminacion)
    numContamination = floor(porcentajeContaminacion/100*size(X,1));
    vectorCentroids = zeros(k,size(X,2),epocs);
    realCentroids = get_centroids_real(labels,X);
    vectorCentroidsContamination = zeros(k,size(X,2),epocs);
    biasVecInitial = zeros(epocs,1);

    biasGlobal = zeros(porcentajeContaminacion,1);
    bias = zeros(porcentajeContaminacion,1);
    biasContamination = zeros(porcentajeContaminacion,1);
end


function [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = nmfAlgorithm(X,k,option)
            [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = NMFCluster(X',k,option);
            contRecur = 0;
            while length(unique(labelsPred)) ~= k
                [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = NMFCluster(X',k,option);
                clear energy;
                if contRecur >30
                    break
                end
                contRecur  = contRecur +1;
            end
end
