function H = kernel_kmeans(H, K, maxepochs)

    if size(H, 1) == 1
        H = ones(size(H))*(1/length(H));
        return
    end

    H = diag(1./(sum(H, 2)))*H;
    [latent_topics, num_samples] = size(H);
    for i = 1:maxepochs
        dist = repmat(sum(H.*(H*K), 2), 1, num_samples) - 2*H*K; 
        [min_dist idx] = min(dist, [], 1);

        [b, ix] = sort(min_dist, 'ascend');

        id = 1
        for j = 1:latent_topics
            if sum(idx == j) == 0
                idx(ix(id)) = j
                id = id+1
            end
        end

        H = full(sparse(idx, 1:num_samples, ones(1, num_samples), latent_topics, num_samples ));
        H = diag(1./(sum(H, 2)))*H;
    end
end
