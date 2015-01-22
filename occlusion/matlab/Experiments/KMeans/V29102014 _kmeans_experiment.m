function correct = kmeans_experiment(filedata,dataset,datalabels,fileresults)

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')



addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

% addpath '/home/jagallegom/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';

k = length(unique(labels));
X= data;
numContamination = floor(0.3*size(X,1));
vectorCentroids = zeros(k,size(X,2),epocs);
realCentroids = get_centroids(labels,X);
vectorCentroidsContamination = zeros(k,size(X,2),epocs);
biasVecInitial = zeros(epocs,1)

biasGlobal = zeros(numContamination,1);
bias = zeros(numContamination,1);
biasContamination = zeros(numContamination,1);

cont = 1

biasAux = Inf;
for i=1:epocs
    try        
        k = length(unique(labels))
%         K = linearKernel(X,X);
%         [labels_pred, energy] = knkmeans(K, k);
        [labelsPred,C]=kmeans(X,k,'Emptyaction','singleton');
        clear energy;
        contRecur = 0;
        while length(unique(labelsPred)) ~= k
            [labelsPred, energy] = kmeans(X,k,'Emptyaction','singleton');
            clear energy;
            if contRecur >30
                break
            end
            contRecur  = contRecur +1
        end
        labelsPred = bestMap(labels,labelsPred);
        vectorCentroids(:,:,i) = get_centroids(labelsPred,X)
        biasVecInitial(i) = sum(sum(abs(vectorCentroids(:,:,i) - realCentroids)));
        if biasVecInitial(i) < biasAux
            labelsReal = labelsPred
            biasAux = biasVecInitial(i)
        end
    catch

    end
    cont = cont + 1
end
[val,pos] = min(biasVecInitial)
centroid = vectorCentroids(:,:,pos);
centroidInitial = centroid;

biasInitial = sum(sum(abs(centroid - realCentroids)));

minFeature = min(X);
maxFeature = max(X);

vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));
    
for j= 1:numContamination
    vectorCentroids = zeros(k,size(X,2),epocs);
    for i=1:epocs       
            k = length(unique(labels))
%             K = linearKernel([X;vector_contamination(1:j,:)],[X;vector_contamination(1:j,:)]);
%             [labels_pred, energy] = knkmeans(K, k);
            labelsPred=kmeans([X;vectorContamination(1:j,:)],k,'Emptyaction','singleton');
            clear energy;
            contRecur = 0;
            while length(unique(labelsPred(1:size(X,1)))) ~= k
%                 [labels_pred, energy] = knkmeans(K, k);
                labelsPred=kmeans([X;vectorContamination(1:j,:)],k,'Emptyaction','singleton');
                clear energy;
                if contRecur >30
                    break
                end
                contRecur  = contRecur +1
            end
            vectorCentroids(:,:,i) = get_centroids(labelsPred(1:size(X,1)),X)
            [labelsPred,vectorCentroids(:,:,i)] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),vectorCentroids(:,:,i));
            vectorCentroidsContamination(:,:,i) = get_centroids(labelsPred,[X;vectorContamination(1:j,:)])
            [labelsPred,vectorCentroidsContamination(:,:,i)] = bestMapNMF(labelsReal,labelsPred(1:size(X,1)),vectorCentroidsContamination(:,:,i));
    end
    centroid = mean(vectorCentroids,3);
    centroidContamination = mean(vectorCentroidsContamination,3);
    biasGlobal(j) = sum(sum(abs(centroid - realCentroids)));
    bias(j) = sum(sum(abs(centroid - centroidInitial)));
    biasContamination(j) =  sum(sum(abs(centroidContamination - centroidInitial)));
end


save(fileresults,'vectorCentroids','biasInitial','bias','realCentroids','X','biasGlobal','vectorContamination','biasContamination','vectorCentroidsContamination')
end


function centroids = get_centroids(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels))
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end

function centroids = mean_of_centroids(vector_centroids)
    k = size(vector_centroids,2)
    centroids = zeros(k,size(vector_centroids,3))
    for i=1:k
        centroids(i,:)= mean(vector_centroids(:,i,:))
    end
end
