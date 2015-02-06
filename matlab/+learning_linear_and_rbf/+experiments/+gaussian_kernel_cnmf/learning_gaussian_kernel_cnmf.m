function correct = learning_gaussian_kernel_cnmf(options)

    import nmf.*
    import utils.*
    import preprocessing_data.*

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

    if strcmp(options.preprocessing,'l1_norm')
        X= L1norm(data);
    end
    if strcmp(options.preprocessing,'normalize_by_range')
        X= normalizeByRange(data,1);
    end
    if strcmp(options.preprocessing,'l2_norm')
        X= L2norm(data,1);
    end

    if strcmp(options.algorithm.name,'kernel_convex_nmf') | strcmp(options.algorithm.name,'kernel_kmeans')
        response = generate_logarithm_vector_kernel(X, options.vect, options.algorithm, 0.5)
        options.vect = [response]
    end
    
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
        options.algorithm.param = options.vect(l)
        for i=1:options.epocs
                tic
                labels_pred = unsupervised_technique(X',options.k,options.algorithm,options.epocs);
                toc
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

function labels_pred = unsupervised_technique(X,k,option,epocs)
    import nmf.*
    import utils.*
    import preprocessing_data.*
    tic
	switch option.name 
		case 'kernel_convex_nmf'
        		labels_pred = kernelconvexnmfCluster(X,k,option);
			toc
			aux2 = 1;
			while length(unique(labels_pred)) ~= k
			    labels_pred = kernelconvexnmfCluster(X,k,option);
			    aux2 = aux2+1
			    if aux2 >epocs
                    break
			    end
			end
		case 'convex_nmf'
        		labels_pred = kernelconvexnmfCluster(X,k,option);
			toc
			aux2 = 1;
			while length(unique(labels_pred)) ~= k
			    labels_pred = kernelconvexnmfCluster(X,k,option);
			    aux2 = aux2+1
			    if aux2 >epocs
                    break
			    end
			end
		case 'kernel_kmeans'
            K = kernelRBF(X,X,option.param);
            labels_pred = knkmeans(K,k);
			toc
			aux2 = 1;
			while length(unique(labels_pred)) ~= k
			    labels_pred = knkmeans(K,k);
			    aux2 = aux2+1
			    if aux2 >epocs
                    break
			    end
			end
		case 'kmeans'
        		labels_pred = kmeans(X,k,'Emptyaction','singleton');
			toc

		case 'nmf'
        		labels_pred = NMFCluster(X,k,option);
			toc

		case 'rmnmf'
        		[labels_pred] = RMNMF(X,labels,k,10^-3,parameters(l,1), parameters(l,2));
			toc
			aux2 = 1;
			while length(unique(labels_pred)) ~= k
			    [labels_pred] = RMNMF(X,labels,k,10^-3,parameters(l,1), parameters(l,2));
			    aux2 = aux2+1
			    if aux2 >epocs
                    break
			    end
            end
    end
end 




