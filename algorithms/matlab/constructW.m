function W = constructW(fea,options) 
%	Usage: 
%	W = constructW(fea,options) 
% 
%	fea: Rows of vectors of data points. Each row is x_i 
%   options: Struct value in Matlab. The fields in options that can be set: 
%           Metric -  Choices are: 
%               'Euclidean' - Will use the Euclidean distance of two data  
%                             points to evaluate the "closeness" between  
%                             them. [Default One] 
%               'Cosine'    - Will use the cosine value of two vectors 
%                             to evaluate the "closeness" between them. 
%                             A popular similarity measure used in 
%                             Information Retrieval. 
%                   
%           NeighborMode -  Indicates how to construct the graph. Choices 
%                           are:  
%                'KNN'            -  Put an edge between two nodes if and 
%                                    only if they are among the k nearst 
%                                    neighbors of each other. You are 
%                                    required to provide the parameter k in 
%                                    the options. [Default One] 
%               'epsilonNeighbor' -  Node i and j will be connected by an 
%                                    edge if  
%                                    'Euclidean': norm(x_i - x_j) < epsilon 
%                                    'Cosine': cosine(x_i,x_j) > epsilon 
%                                    You are required to provide the 
%                                    parameter epsilon in the options. 
%               'Supervised'      -  Put an edge between two nodes if and 
%                                    only if they belong to same class. You 
%                                    are required to provide the label 
%                                    information gnd in the options. 
%                                               
%           WeightMode   -  Indicates how to assign weights for each edge 
%                           in the graph. Choices are: 
%               'Binary'       - 0-1 weighting. Every edge receiveds weight 
%                                of 1. [Default One] 
%               'HeatKernel'   - If nodes i and j are connected, put weight 
%                                W_ij = exp(-norm(x_i - x_j)/t). This 
%                                weight mode can only be used under 
%                                'Euclidean' metric and you are required to 
%                                provide the parameter t. 
%               'Cosine'       - If nodes i and j are connected, put weight 
%                                cosine(x_i,x_j). Can only be used under 
%                                'Cosine' metric. 
%                
%            k         -   The parameter needed under 'KNN' NeighborMode. 
%                          Default will be 5. 
%            epsilon   -   The parameter needed under 'epsilonNeighbor' 
%                          NeighborMode. Default will be 0.5 
%            gnd       -   The parameter needed under 'Supervised' 
%                          NeighborMode.  Colunm vector of the label 
%                          information for each data point. 
%            bLDA      -   0 or 1. Only effective under 'Supervised' 
%                          NeighborMode. If 1, the graph will be constructed 
%                          to make LPP exactly same as LDA. Default will be 
%                          0.  
%            t         -   The parameter needed under 'HeatKernel' 
%                          WeightMode. Default will be 1 
%         bNormalized  -   0 or 1. Only effective under 'Cosine' metric. 
%                          Indicates whether the fea are already be 
%                          normalized to 1. Default will be 0 
%      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 1 
%                          if 'Supervised' NeighborMode & bLDA == 1, 
%                          bSelfConnected will always be 1. 
% 
% 
%    Examples: 
% 
%       fea = rand(50,15); 
%       options = []; 
%       options.Metric = 'Euclidean'; 
%       options.NeighborMode = 'KNN'; 
%       options.k = 5; 
%       options.WeightMode = 'HeatKernel'; 
%       options.t = 1; 
%       W = constructW(fea,options); 
%        
%        
%       fea = rand(50,15); 
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4]; 
%       options = []; 
%       options.Metric = 'Euclidean'; 
%       options.NeighborMode = 'Supervised'; 
%       options.gnd = gnd; 
%       options.WeightMode = 'HeatKernel'; 
%       options.t = 1; 
%       W = constructW(fea,options); 
%        
%        
%       fea = rand(50,15); 
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4]; 
%       options = []; 
%       options.Metric = 'Euclidean'; 
%       options.NeighborMode = 'Supervised'; 
%       options.gnd = gnd; 
%       options.bLDA = 1; 
%       W = constructW(fea,options);       
%        
 
%    For more details about the different ways to construct the W, please 
%    refer: 
%       Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using 
%       Locality Preserving Indexing" IEEE TKDE, Dec. 2005. 
%     
% 
%    Written by Deng Cai (dengcai@gmail.com), April/2004, Feb/2006 
%  
 
if (~exist('options','var')) 
   options = []; 
else 
   if ~strcmpi(class(options),'struct')  
       error('parameter error!'); 
   end 
end 
 
%================================================= 
if ~isfield(options,'Metric') 
    options.Metric = 'Euclidean'; 
end 
 
switch lower(options.Metric) 
    case {lower('Euclidean')} 
        ; 
    case {lower('Cosine')} 
        if ~isfield(options,'bNormalized') 
            options.bNormalized = 0; 
        end 
    otherwise 
        error('Metric does not exist!'); 
end 
 
%================================================= 
if ~isfield(options,'NeighborMode') 
    options.NeighborMode = 'KNN'; 
end 
 
switch lower(options.NeighborMode) 
    case {lower('KNN')}  %For simplicity, we include the data point itself in the kNN 
        if ~isfield(options,'k') 
            options.k = 5; 
        end 
        if options.k < 1 
            options.k = 1; 
        end 
    case {lower('epsilonNeighbor')} 
        if ~isfield(options,'epsilon') 
            options.epsilon = 0.5; 
        end 
    case {lower('Supervised')} 
        if ~isfield(options,'bLDA') 
            options.bLDA = 0; 
        end 
        if options.bLDA 
            options.bSelfConnected = 1; 
        end 
        if ~isfield(options,'gnd') 
            error('Label(gnd) should be provided under ''Supervised'' NeighborMode!'); 
        end 
        if length(options.gnd) ~= size(fea,1) 
            error('gnd doesn''t match with fea!'); 
        end 
    otherwise 
        error('NeighborMode does not exist!'); 
end 
 
%================================================= 
 
if ~isfield(options,'WeightMode') 
    options.WeightMode = 'Binary'; 
end 
 
switch lower(options.WeightMode) 
    case {lower('Binary')} 
        ; 
    case {lower('HeatKernel')} 
        if ~strcmpi(options.Metric,'Euclidean') 
            warning('''HeatKernel'' WeightMode should be used under ''Euclidean'' Metric!'); 
            options.Metric = 'Euclidean'; 
        end 
        if ~isfield(options,'t') 
            options.t = 1; 
        end 
    case {lower('Cosine')} 
        if ~strcmpi(options.Metric,'Cosine') 
            warning('''Cosine'' WeightMode should be used under ''Cosine'' Metric!'); 
            options.Metric = 'Cosine'; 
        end 
        if ~isfield(options,'bNormalized') 
            options.bNormalized = 0; 
        end 
    otherwise 
        error('WeightMode does not exist!'); 
end 
 
%================================================= 
 
if ~isfield(options,'bSelfConnected') 
    options.bSelfConnected = 1; 
end 
 
%================================================= 
[nSmp, nFea] = size(fea); 
 
 
if strcmpi(options.NeighborMode,'Supervised') & (options.bLDA | strcmpi(options.WeightMode,'Binary')) 
    ; 
else 
    bDistance = 0; 
    if strcmpi(options.Metric,'Euclidean') 
        D = zeros(nSmp); 
        for i=1:nSmp-1 
            for j=i+1:nSmp 
                D(i,j) = norm(fea(i,:) - fea(j,:)); 
            end 
        end 
        D = D+D'; 
        bDistance = 1; 
    else 
        if options.bNormalized 
            D = fea * fea'; 
        else 
            feaNorm = sum(fea.^2,2).^.5; 
            fea = fea ./ repmat(max(1e-10,feaNorm),1,size(fea,2)); 
            D = fea * fea'; 
        end 
    end 
end 
 
 
switch lower(options.NeighborMode) 
    case {lower('KNN')} 
        if options.k >= nSmp 
            G = ones(nSmp,nSmp); 
        else 
            G = zeros(nSmp,nSmp); 
            if bDistance 
                [dump idx] = sort(D, 2); % sort each row 
            else 
                [dump idx] = sort(-D, 2); % sort each row 
            end 
            for i=1:nSmp 
                G(i,idx(i,1:options.k+1)) = 1; 
            end 
        end 
    case {lower('epsilonNeighbor')} 
        if bDistance 
            [i,j] = find(D < options.epsilon); 
        else 
            [i,j] = find(D > options.epsilon); 
        end 
        G = sparse(i,j,1); 
    case {lower('Supervised')} 
        G = zeros(nSmp,nSmp); 
 
        Label = unique(options.gnd); 
        nLabel = length(Label); 
        if options.bLDA 
            for idx=1:nLabel 
                classIdx = find(options.gnd==Label(idx)); 
                G(classIdx,classIdx) = 1/length(classIdx); 
            end 
            W = sparse(G); 
            return; 
        else 
            for idx=1:nLabel 
                classIdx = find(options.gnd==Label(idx)); 
                G(classIdx,classIdx) = 1; 
            end 
        end 
         
        if strcmpi(options.WeightMode,'Binary') 
            if ~options.bSelfConnected 
                G  = G - diag(diag(G)); 
            end 
            W = sparse(G); 
            return; 
        end 
    otherwise 
        error('NeighborMode does not exist!'); 
end 
 
if ~options.bSelfConnected 
    G  = G - diag(diag(G)); 
end 
 
switch lower(options.WeightMode) 
    case {lower('Binary')} 
        W = max(G,G'); 
        W = sparse(W); 
    case {lower('HeatKernel')} 
        D = exp(-D.^2/options.t); 
        W = D.*G; 
        W = max(W,W'); 
        W = sparse(W); 
    case {lower('Cosine')} 
        W = D.*G; 
        W = max(W,W'); 
        W = sparse(W); 
    otherwise 
        error('WeightMode does not exist!'); 
end 