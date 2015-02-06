function matrix = contigency_matrix(labels_true, labels_pred)
    [LT,LTia,LTic] = unique(labels_true);
    [LP,LPia,LPic] = unique(labels_pred);
    clear LTic
    clear LPic
    matrix = zeros(length(LT),length(LP));
    for i = 1:length(LT)
        for j = 1:length(LP)
            matrix(i,j) = sum(labels_true(labels_pred == labels_pred(LPia(j))) == labels_true(LTia(i)));
        end
    end 
end