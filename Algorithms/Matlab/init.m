function H = init(K,pos)
    if length(pos) == 1
        H = ones(size(K,2),1);
        return
    end
    num_objects = size(K,2);
    LT = length(pos);
    D = diag(K(pos,pos))*ones(1,num_objects) + ones(LT,1)*diag(K)' - 2*K(pos,:);
    [num idx] = min(D);
    clear num;
    H = full(sparse(1:num_objects, idx, ones(1, num_objects),num_objects,LT));
end
