function vect  = generate_logarithm_vector_kernel(real_matrix, vect_to_prove, options, percentage)
    
    import nmf.computeKernelMatrix
%Generate a vector with according to a kernel such that
%        the computed kernel have the 99% of the data greater or
%        less than 0 and the 100% of the data aren't equal to 1
%----------
%        real_matrix: n_samples x n_features
%        options: dictionary
%
%        Returns
%        -------
%        vector variable

%   """
    vect = [];

    for i = 1:length(vect_to_prove)
        %# We have to correct this line to change with factory kernel
        options.param = vect_to_prove(i);
        computed_kernel = computeKernelMatrix( real_matrix',real_matrix',options);
        if(sum(sum(computed_kernel > 0 & computed_kernel < 1)) / numel(computed_kernel) > percentage);
            vect = [vect vect_to_prove(i)];
        end
    end

end