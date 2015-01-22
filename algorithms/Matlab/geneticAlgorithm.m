function centroidsEstimate = geneticAlgorithm(centroidsReal,centroidsEstimate)
    numFathers = 80;
    numSons = 80;
    probMutacion = 0.01;
    numIter = 20;
    
    classes = size(centroidsReal,1);
    vectLink = 1:classes;
    
    vectFathers = fathersInitializing(vectLink,numFathers);
    vectAptitude = getAttitude(centroidsReal,centroidsEstimate,vectFathers);
    
    for k = 1:numIter
        n = length(vectLink);
        vectSons = zeros(numSons, n);
        for j = 1:numSons
            selection = fathersSelection(vectAptitude);
            vectSons(j,:) = vectFathers(selection,:);
            if rand()<=probMutacion
                for i=1: randi([0,3])
                    shuffle=randperm(n);
                    shuffle=shuffle(1:2);
                    aux = vectSons(j,shuffle(1));
                    vectSons(j,shuffle(1)) = vectSons(j,shuffle(2));
                    vectSons(j,shuffle(2)) = aux;
                end
            end
        end
        vectFathers  = vectSons;
        vectAptitude = getAttitude(centroidsReal,centroidsEstimate,vectFathers);
    end
    [~, index] = min(vectAptitude);
    centroidsEstimate = centroidsEstimate(vectFathers(index,:),:);

end

function vectFathers = fathersInitializing(vectLink,numFathers)
    vectFathers = zeros(numFathers, length(vectLink));
    for i=1:numFathers
       vectFathers(i,:) = vectLink(randperm(length(vectLink)));
    end

end

function vectAptitude = getAttitude(centroidsReal,centroidsEstimate,vectLink)
    n = size(vectLink,1);
    vectAptitude = zeros(1,n);
    for i=1:n
           vectAptitude(i) = norm(centroidsReal - centroidsEstimate(vectLink(i,:),:),'fro');
    end
end

function selection = fathersSelection(vectAptitude)
    vectAptitudeAccumulate = calcuateAccumulateProbability(vectAptitude);
    number = rand();
    if number<=vectAptitudeAccumulate(1)
        selection = 1;
    else
        for i=2:length(vectAptitudeAccumulate)
             if number<=vectAptitudeAccumulate(i) && number(1)>vectAptitudeAccumulate(i-1) 
                 selection = i;
             end
        end
    end
end

function vectAptitudeAccumulate = calcuateAccumulateProbability(vectAptitude)
    vectAptitudeAux = min(vectAptitude) - vectAptitude + max(vectAptitude);
    aux = 0;
    sumAptitude = sum(vectAptitudeAux);
    vectAptitudeAccumulate = zeros(size(vectAptitudeAux));
    for i=1:length(vectAptitudeAux)
        aux = vectAptitudeAux(i) + aux;
        vectAptitudeAccumulate(i) = aux / sumAptitude;
    end
end
