function accuracy = clusteringaccuracy(labels_true,labels_pred)
    matrix= contigency_matrix(labels_true,labels_pred);
    accuracy = get_indicesClusterXClass(matrix);
    clear matrix
end
function contigency = contigency_matrix(labels_true,labels_pred)
    classes = max(size(unique(labels_true)));
    clusters = max(size(unique(labels_pred)));
    unique_labels_true = unique(labels_true)
    unique_labels_pred = unique(labels_pred)
    contigency = zeros(classes,clusters);
    for i=1:classes
        for j= 1:clusters
            contigency(i,j) = sum(labels_true(labels_pred == unique_labels_pred(j))==unique_labels_true(i));
        end
    end
    max(size(unique(labels_pred)))
    sum(sum(contigency))
    clear classes;
    clear clusters;
end
function accuracy = get_indicesClusterXClass(matrix)
    cost_matrix = zeros(size(matrix));
    for row=1:size(matrix,1)
        for col=1:size(matrix,2)
           cost_matrix(row,col) = max(max(matrix)) - matrix(row,col);
        end        
    end
       
    [assignment,cost] = munkres(cost_matrix);
    clear cost;
    clear cost_matrix;
    sum(sum(matrix))
    accuracy = sum(sum(matrix(assignment)) )/ sum(sum(matrix));
end