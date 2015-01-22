function X = L1norm(X)
    X = X./( ones(size(X)) * diag(sum(abs(X))) );
