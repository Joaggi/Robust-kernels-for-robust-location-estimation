

function correct = learning_nmf(options)

    import nmf.*
    import utils.*

    c = clock;
    nameresults = strcat(num2str(c(1)),num2str(c(2)),num2str(c(3)),'_',num2str(c(4))...
        ,'-',num2str(c(5)));

    load(options.dataset.name,'-mat', options.dataset.dataset, options.dataset.labels)

    data = eval(options.dataset.dataset);
    labels = eval(options.dataset.labels);
    
    %nu = randsample(size(data,1),2000)

    %data = data(nu,:);
    %labels = labels(nu,:);

    options.k = length(unique(labels));
    X = data;
    % X= L1norm(data);
    % X = X'
    
    if strcmp(options.preprocessing,'normalize_by_range')
        X= normalizeByRange(data,1);
    end

    cont = 1

    purityVec = zeros(options.epocs,1);
    clusteringAccuracyVec = zeros(options.epocs,1);
    nmiVec = zeros(options.epocs,1);
    labels_total = zeros(options.epocs,length(labels));

    cont = 1

    for i=1:options.epocs
            k = length(unique(labels));
            labels_pred = NMFCluster(X',options.k,options.algorithm);
            labels_total(cont,:) = labels_pred;     
            [result,~] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1);
            nmiVec(cont) = result(2);
            purityVec(cont) = result(3);
        cont = cont + 1
    end

    purityMeanVec = mean(purityVec)
    clusteringAccuracyMeanVec = mean(clusteringAccuracyVec)
    nmiMeanVec = mean(nmiVec)
    puritySdVec= std(purityVec)
    clusteringAccuracySdVec = std(clusteringAccuracyVec)
    nmiSdVec = std(nmiVec)

    save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
