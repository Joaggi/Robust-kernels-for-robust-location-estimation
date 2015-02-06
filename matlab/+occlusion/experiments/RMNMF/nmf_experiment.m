function correct = nmf_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end


addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

epocs = 150

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';

% addpath '/home/jagallegom/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';

vectLambda = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3]
rho = [1.05,1.15,1.25,1.35,1.45,1.66,1.8,2.2,2.5,3]

[X,Y] = meshgrid(vectLambda,rho);
parameters = [X(:) Y(:)];

k = length(unique(labels));
%X= L1norm(data);
% X = data;
X= normalizeByRange(data,1);



porcentajeContaminacion= 30
%Definicion de variables
[numContamination,vectorCentroids,realCentroids,vectorCentroidsContamination,bias,biasContamination,biasVecInitial,biasGlobal] =   definicionVariables(X,labels,k,epocs,porcentajeContaminacion);

cont = 1

option.reorder=false;
option.algorithm='nmfrule';
option.iterations=2000;
option.iter=2000;
option.distance = 'ls';
option.dis = false;

biasAux = Inf;
for l = 1:num
    aux=cont
    for i=1:epocs
        aux = cont
        for i=1:epocs
            addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';
            k = length(unique(labels));
            [labels_pred,Yout] = RMNMF(X',labels,k,10^-3,parameters(l,1), parameters(l,2));
             addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
            vectorCentroids(:,:,i) = get_centroids(labelsPred,X,Yout);            
            [labelsPred,vectorCentroids(:,:,cont)] = bestMapKMeans(labels,labelsPred,vectorCentroids(:,:,cont));
            cont = cont + 1
        end
        centroids(:,:,l) = mean(vectorCentroids(:,:,(l-1)*epocs+1:(l)*epocs),3);
        bias(l) = sum(sum(abs(centroids(:,:,l) - realCentroids)));
        if bias(l) < biasAux
            labelsReal = labelsPred;
            biasAux = bias(l);
        end
           
            
            
            biasVecInitial(i) = sum(sum(abs(vectorCentroids(:,:,i) - realCentroids)));
            if biasVecInitial(i) < biasAux
                labelsReal = labelsPred;
                biasAux = biasVecInitial(i);
            end
            cont = cont + 1
    end
    centroid = mean(vector_centroids,3);
    bias_aux = sum(sum(centroid - real_centroids));
    if bias_aux < bias_initial
        bias_initial = bias_aux
        parameters_best = parameters(l,:)
    end
    [s,centroid,centroidInitial,biasInitial] = getResults(bias,centroids,realCentroids,vect);
end

centroid = mean(vectorCentroids,3);
centroidInitial = centroid;

biasInitial = sum(sum(abs(centroid - realCentroids)));

minFeature = min(X)-min(X)*0.5;
maxFeature = max(X)+max(X)*0.5;

vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));
     

for j= 1:porcentajeContaminacion
    vectorCentroidsContamination = zeros(k,size(X,2),epocs);
    booleanVectorCentroids = ones(1,epocs);
    XwithContamination = [X;vectorContamination(1:floor(size(X,1)*(j/100)),:)];
        for i=1:epocs      
            try
                k = length(unique(labels))
                [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = nmfAlgorithm(XwithContamination,k,option);
                %[labelsPred,Yout] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),Yout);
                vectorCentroidsContamination(:,:,i) = get_centroids(labelsPred,XwithContamination,Yout);
                [labelsPred,vectorCentroidsContamination(:,:,i)] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),vectorCentroidsContamination(:,:,i));
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


save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','biasGlobal','vectorContamination','biasContamination','vectorCentroidsContamination')
end


function centroids = get_centroids(labels,data,H)
    unique_labels = unique(labels);
    k = length(unique(labels))
    centroids = zeros(k,size(data,2));
   % H=H ./ repmat(sum(H,1),k,1)
    for i=1:k
        centroids(i,:)= sum(repmat(H(i,labels == unique_labels(i))',1,size(data,2)).*data(labels == unique_labels(i),:))/sum(H(i,labels == unique_labels(i)));
    end
end

function centroids = get_centroids_real(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels))
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
    biasVecInitial = zeros(epocs,1)

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
                contRecur  = contRecur +1
            end
end
