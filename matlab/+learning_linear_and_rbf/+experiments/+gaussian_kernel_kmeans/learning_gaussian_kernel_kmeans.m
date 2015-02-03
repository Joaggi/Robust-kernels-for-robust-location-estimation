function correct = learning_gaussian_kernel_kmeans(options)

    import nmf.*
    import utils.*

    c = clock;
    nameresults = strcat(num2str(c(1)),num2str(c(2)),num2str(c(3)),'_',num2str(c(4))...
        ,'-',num2str(c(5)));

    load(options.dataset.name,'-mat', options.dataset.dataset, options.dataset.labels)

    data = eval(options.dataset.dataset);
    labels = eval(options.dataset.labels);
    
    nu = randsample(size(data,1),2000)

    data = data(nu,:);
    labels = labels(nu,:);

    options.k = length(unique(labels));
    X = data;
    % X= L1norm(data);
    % X = X'
    % X= normalizeByRange(data,1);

    options.vect = generate_logarithm_vector_kernel(X, options.vect, options.algorithm ...
        , 0.5)

    cont = 1


    purityVec = zeros(options.epocs*length(options.vect),1);
    clusteringAccuracyVec = zeros(options.epocs*length(options.vect),1);
    nmiVec = zeros(options.epocs*length(options.vect),1);
    labels_total = zeros(options.epocs*length(options.vect),length(labels));

    purityMeanVec = zeros(length(options.vect),1);
    clusteringAccuracyMeanVec = zeros(length(options.vect),1);
    nmiMeanVec = zeros(length(options.vect),1);

    puritySdVec = zeros(length(options.vect),1);
    clusteringAccuracySdVec = zeros(length(options.vect),1);
    nmiSdVec = zeros(length(options.vect),1);


    l = 1

    while(l<=length(options.vect))
        aux = cont
        options.algorithm.param = options.vect(l);
        K = kernelRBF(X',X',options.vect(l));
        for i=1:options.epocs
                tic
                [labels_pred,~] = knkmeans(K, options.k);
                toc
                aux2 = 1;
                while length(unique(labels_pred)) ~= options.k
                    labels_pred = kernelconvexnmfCluster(X',options.k,options.algorithm);
                    aux2 = aux2+1
                    if aux2 >options.epocs
                        break
                    end
                end
                labels_total(cont,:) = labels_pred;
                [result,~] = ClusteringMeasure(labels,labels_pred)
                clusteringAccuracyVec(cont) = result(1);
                purityVec(cont) = result(3);
                nmiVec(cont) = result(2);
            cont = cont + 1
        end
        purityMeanVec(l) = mean(purityVec(aux:cont-1));
        clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
        nmiMeanVec(l) = mean(nmiVec(aux:cont-1));
        puritySdVec(l) = std(purityVec(aux:cont-1));
        clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1));
        nmiSdVec(l) = std(nmiVec(aux:cont-1));
        l=l+1

    end
    save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','l','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end




