function X = normalizeByRange(X,axis)
    if axis == 1
        X = X - repmat(min(X,[],1),size(X,1),1);
        rango = repmat(max(X,[],1),size(X,1),1) - repmat(min(X,[],1),size(X,1),1);
    end
    if axis == 2
        X = X - repmat(min(X,[],2),1,size(X,2));
        rango = repmat(max(X,[],2),1,size(X,2)) - repmat(min(X,[],2),1,size(X,2));
    end
    X = X ./ rango;
end