function [X,minimo,maximo] = normalizeByRange(X,axis,minFeature,maxFeature)
    minimo = min(X,[],1);
    maximo = max(X,[],1);
    if (nargin == 2)
        if(axis == 1)
            X = X - repmat(min(X,[],1),size(X,1),1);
            rango = repmat(max(X,[],1),size(X,1),1) - repmat(min(X,[],1),size(X,1),1);
        end
        if(axis == 2)
            X = X - repmat(min(X,[],2),1,size(X,2));
            rango = repmat(max(X,[],2),1,size(X,2)) - repmat(min(X,[],2),1,size(X,2));
        end
        X = X ./ rango;
    end
	if(nargin == 4)
	    if(axis == 1)
	        X = X - repmat(minFeature,size(X,1),1);
	        rango = repmat(maxFeature,size(X,1),1) - repmat(minFeature,size(X,1),1);
	    end
	    if(axis == 2)
	        X = X - repmat(minFeature,1,size(X,2));
	        rango = repmat(maxFeature,1,size(X,2)) - repmat(minFeature,1,size(X,2));
	    end
	    X = X ./ rango;		
end