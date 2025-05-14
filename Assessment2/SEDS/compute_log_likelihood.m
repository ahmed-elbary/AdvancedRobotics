function loglik = compute_log_likelihood(Data, model)
    nbStates = model.nbStates;
    [~, nbData] = size(Data);
    Pxi = zeros(nbStates, nbData);
    for i = 1:nbStates
        Pxi(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
    end
    loglik = sum(log(sum(Pxi,1) + realmin));
end
