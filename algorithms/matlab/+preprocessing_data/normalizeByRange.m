function [real_matrix,contaminated_matrix] = normalizeByRange(real_matrix,axis,contaminated_matrix)

    if (nargin == 2)
        if(axis == 1)
            minimo_1 = min(real_matrix,[],1);
            maximo_1 = max(real_matrix,[],1);
            real_matrix = real_matrix - repmat(minimo_1,size(real_matrix,1),1);
            rango = repmat(maximo_1,size(real_matrix,1),1) - repmat(minimo_1,size(real_matrix,1),1);
        end
        if(axis == 2)
            minimo_2 = min(real_matrix,[],1);
            maximo_2 = max(real_matrix,[],1);
            real_matrix = real_matrix - repmat(minimo_2,1,size(real_matrix,2));
            rango = repmat(maximo_2,1,size(real_matrix,2)) - repmat(minimo_2,1,size(real_matrix,2));
        end
        real_matrix = real_matrix ./ rango;
    end
	if(nargin == 3)
	    if(axis == 1)
	        contaminated_matrix = contaminated_matrix - repmat(minimo_1,size(real_matrix,1),1);
	        rango = repmat(maximo_1,size(real_matrix,1),1) - repmat(minimo_1,size(real_matrix,1),1);
	    end
	    if(axis == 2)
	        contaminated_matrix = contaminated_matrix - repmat(minimo_2,1,size(real_matrix,2));
	        rango = repmat(maximo_2,1,size(real_matrix,2)) - repmat(minimo_2,1,size(real_matrix,2));
	    end
	    contaminated_matrix = real_matrix ./ rango;		
end