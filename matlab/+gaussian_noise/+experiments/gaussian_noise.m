function kernelconvexnmf_experiment(options)

    import nmf.*
    import utils.*
    import preprocessing_data.*
    import dataset.*

    c = clock;
    nameresults = strcat(options.dataset.name, '_', options.algorithm.name, '_', options.preprocessing, '_', num2str(c(1)), num2str(c(2)), num2str(c(3)),'_',num2str(c(4))...
        ,'-',num2str(c(5)));

    load(options.dataset.name,'-mat', options.dataset.dataset, options.dataset.labels, options.dataset.dataset_contamination)

    data_real = double(eval(options.dataset.dataset));
    data_contamination = double(eval(options.dataset.dataset_contamination));
    labels = double(eval(options.dataset.labels));
    
    %nu = randsample(size(data,1),2000)

    %data = data(nu,:);
    %labels = labels(nu,:);

    options.k = length(unique(labels));
    X = data_real;

    if strcmp(options.preprocessing,'l1_norm')
        [X,data_contamination]= L1norm(data_real);
    end
    if strcmp(options.preprocessing,'normalize_by_range')
        [X,data_contamination]= normalizeByRange(data_real,1,data_contamination);
    end
     if strcmp(options.preprocessing,'l2_norm')
        [X,data_contamination]= L2norm(data_real,1);
    end

    if strcmp(options.algorithm.name,'kernel_convex_nmf') | strcmp(options.algorithm.name,'kernel_kmeans')
        response = generate_logarithm_vector_kernel(X, options.vect, options.algorithm, 0.5)
        options.vect = [response]
    end
    
    cont = 1

    clusteringAccuracyVec = zeros(options.epocs*length(options.vect),1);
    clusteringAccuracyMeanVec = zeros(length(options.vect),1);
    clusteringAccuracySdVec = zeros(length(options.vect),1);


    porcentajeContaminacion= 30;

    cont = 1;
    for l = 1:length(options.vect)
        aux = cont;
        options.algorithm.param = options.vect(l)
        for i=1:options.epocs       
                labels_pred = unsupervised_technique(X',options.k,options.algorithm,options.epocs);
                [result,~] = ClusteringMeasure(labels,labels_pred);
                clusteringAccuracyVec(cont) = result(1)
                cont = cont + 1
        end
        clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
        clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1));
    end


    [~, posMin] = max(clusteringAccuracyMeanVec);
    s = options.vect(posMin);

    options.algorithm.param= options.vect(posMin);
    [numContamination,clusteringAccuracyVecContamination,clusteringAccuracyMeanVecContamination,clusteringAccuracySdVecContamination] =   definicion_variables_contamination(X,options.vect,options.epocs,porcentajeContaminacion)

    cont = 1;
    j=1;
    while(j<=porcentajeContaminacion)
            aux = cont;
            for i=1:options.epocs      
                XwithContamination = X;
                random = randperm(size(data_contamination,1))
                randompermutation = random(1:floor(size(data_contamination,1)*(j/100)));
                XwithContamination(randompermutation,:) = data_contamination(randompermutation,:);
                k = length(unique(labels));
                [labelsPred] = unsupervised_technique(XwithContamination',options.k,options.algorithm,options.epocs);
                [result,~] = ClusteringMeasure(labels,labelsPred);
                clusteringAccuracyVecContamination(cont) = result(1);
                cont = cont + 1
            end
            clusteringAccuracyMeanVecContamination(j) = mean(clusteringAccuracyVecContamination(aux:cont-1));
            clusteringAccuracySdVecContamination(j) = std(clusteringAccuracyVecContamination(aux:cont-1));        
            j=j+1
    end


    save(nameresults,'clusteringAccuracyVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','clusteringAccuracyVecContamination','clusteringAccuracyMeanVecContamination','clusteringAccuracySdVecContamination','X','j','s','options')
end


function [clusteringAccuracyVec,clusteringAccuracyMeanVec,clusteringAccuracySdVec] =   definicionVariables(X,vect,epocs,porcentajeContaminacion)
    clusteringAccuracyVec = zeros(epocs*length(vect),1);
    clusteringAccuracyMeanVec = zeros(length(vect),1);
    clusteringAccuracySdVec = zeros(length(vect),1);
end
function [numContamination,clusteringAccuracyVecContamination,clusteringAccuracyMeanVecContamination,clusteringAccuracySdVecContamination] =   definicion_variables_contamination(X,vect,epocs,porcentajeContaminacion)
    numContamination = floor(porcentajeContaminacion/100*size(X,1));
    clusteringAccuracyVecContamination = zeros(porcentajeContaminacion*epocs,1);
    clusteringAccuracyMeanVecContamination = zeros(porcentajeContaminacion,1);
    clusteringAccuracySdVecContamination = zeros(porcentajeContaminacion,1);
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
            labels_pred = knkmeans(K,k,option,X);
			toc
			aux2 = 1;
			while length(unique(labels_pred)) ~= k
			    labels_pred = knkmeans(K,k,option,X);
			    aux2 = aux2+1
			    if aux2 >epocs
                    break
			    end
			end
		case 'kmeans'
        		labels_pred = kmeans(X',k,'Emptyaction','singleton');
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





