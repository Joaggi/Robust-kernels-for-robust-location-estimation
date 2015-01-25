function z = purity(labels_true, labels_pred)
    k = max(size(unique(labels_true)));
    purityVector = zeros(1,k);
    purity = 0;
    matrix = contigency_matrix(labels_true, labels_pred);
    for i=1:k
        moda = max(matrix(:,i));
        purityVector(i) = moda / sum(matrix(:,i));
        purity = purityVector(i) * sum(matrix(:,i)) / length(labels_pred) + purity
    end
    clear purityVector;
    clear ind;
    clear moda;
    z = purity ;
end
