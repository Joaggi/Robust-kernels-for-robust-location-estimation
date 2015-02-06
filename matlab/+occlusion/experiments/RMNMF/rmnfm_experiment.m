function correct = rmnmf_experiment(filedata,dataset,datalabels,fileresults)

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')
epocs = 1


vectLambda = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3]
rho = [1.05,1.15,1.25,1.35,1.45,1.66,1.8,2.2,2.5,3]

vectLambda = vectLambda(1)
rho = rho(6)

% vectLambda = [10^-3]
% rho = [1.66]

[X,Y] = meshgrid(vectLambda,rho);
parameters = [X(:) Y(:)];

num = size(parameters,1);



addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab';

k = max(size(unique(labels)));
X= data;
%X = data;
% X= standarization(data');
% X = X'

num_contamination = floor(0.1*size(X,1));
vector_centroids = zeros(k,size(X,2),epocs);
vector_centroids_all = zeros(k,size(X,2),epocs,num);
real_centroids = get_centroids_real(labels,X);

bias = zeros(num_contamination,1);

cont = 1;

bias_initial = Inf
parameters_best = zeros(1,2)


for l = 1:num
    aux=cont
    for i=1:epocs
%         try
            i
            E = X';
            %fprintf('Legue')
            [labels_pred,G] = RMNMF(E,labels,k,10^-3,parameters(l,1), parameters(l,2));
	        labels_total(cont,:) = labels_pred;
            %F = G
            %[labels_pred,G] = bestMapNMF(labels,labels_pred,G');
            %G = G'
            vector_centroids(:,:,i) = get_centroids(labels_pred,X,G');
            vect = vector_centroids(:,:,i)
            [labels_pred,vector_centroids(:,:,i)] = bestMapNMF(labels,labels_pred,vector_centroids(:,:,i));
            vector_centroids_all(:,:,i,l) = vector_centroids(:,:,i);
%         catch
%         end
    end
    centroid = mean(vector_centroids,3);
    bias_aux = sum(sum(centroid - real_centroids));
    if bias_aux < bias_initial
        bias_initial = bias_aux
        parameters_best = parameters(l,:)
    end
end

min_feature = min(X);
max_feature = max(X);

vector_contamination = unifrnd(repmat(min_feature,num_contamination,1),repmat(max_feature,num_contamination,1),num_contamination,length(min_feature));
%     
% for j= 1:num_contamination
%     vector_centroids = zeros(k,size(X,2),epocs);
%     for i=1:epocs       
%             k = length(unique(labels))
%             [labels_pred] = RMNMF(E,labels,k,10^-3,parameters_best(1), parameters_best(2));
%             clear energy;
%             contRecur = 0;
%             while length(unique(labels_pred)) ~= k
%                 [labels_pred, energy] = knkmeans(K, k);
%                 clear energy;
%                 if contRecur >30
%                     break
%                 end
%                 contRecur  = contRecur +1
%             end
%             labels_pred = bestMap(labels,labels_pred(1:size(X,1)),labels_pred);
%             vector_centroids(:,:,i) = get_centroids(labels_pred,[X;vector_contamination(1:j,:)])
%     end
%     centroid = mean(vector_centroids,3);
%     bias(j) = sum(sum(centroid - real_centroids));
% end


save(fileresults,'vector_centroids','bias_initial','real_centroids','X','bias','vector_contamination','vector_centroids_all')     
end


function centroids = get_centroids(labels,data,H)
    unique_labels = unique(labels);
    k = length(unique(labels))
    centroids = zeros(k,size(data,2));
    %H=H ./ repmat(sum(H,1),k,1)
    for i=1:k
        centroids(i,:)= sum(repmat(H(i,labels == unique_labels(i))',1,size(data,2)).*data(labels == unique_labels(i),:));
    end
end

function centroids = mean_of_centroids(vector_centroids)
    k = size(vector_centroids,2)
    centroids = zeros(k,size(vector_centroids,3))
    for i=1:k
        centroids(i,:)= mean(vector_centroids(:,i,:))
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

