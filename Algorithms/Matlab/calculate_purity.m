function [purity,purityVector] = calculate_purity(labels,vectorY, k)
    purityVector = zeros(1,k);
    purity = 0;
    for i=1:k
       [moda,modaNum]=mode(vectorY(find(labels == i)));
       purityVector(i) = modaNum/max(size(find(labels==i)));
    end
    for i=1:k
       purity(1) = purityVector(i)*max(size(find(labels==i))) + purity;
    end
    purity = purity / max(size(labels));
end