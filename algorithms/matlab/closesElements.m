function H = closesElements(K,Z,KZ)
    nrows = size(K,2);
    ncols = size(Z,2);
    if ncols == 1
        H = ones(nrows,ncols);
        return
    end
    D = diag(Z)*ones(1,nrows) + ones(ncols,1)*diag(K)' - 2*KZ';
    [num idx] = min(D);
    clear num;
    H = full(sparse(1:nrows, idx, ones(1, nrows),nrows,ncols));
end
