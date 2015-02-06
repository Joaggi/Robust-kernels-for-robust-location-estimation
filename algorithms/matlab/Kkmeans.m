function [H,idx] = Kkmeans(H,K, maxepochs)
    if size(H,2)==1
         H = ones(size(H))*(1/length(H));
        return
    end
    num_Objects = size(H,1);
    LT = size(H,2);
    dist = zeros(size(H));
    for i = 1: maxepochs
        n = diag(H'*H);
        s = (diag(H'*K*H))./(n.*n);
        dist = s*ones(1,num_Objects) + ones(LT,1)*diag(K)' - 2*diag(1./n)*H'*K;
        [temp idx] = min(dist);
        clear temp;
        H = full(sparse(1:num_Objects, idx, ones(1,num_Objects), num_Objects, LT )); 
    end
    H = H*diag(1./(diag(H'*H)));
