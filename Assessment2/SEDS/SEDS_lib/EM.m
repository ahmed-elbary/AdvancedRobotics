function [Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma)
% EM algorithm for Gaussian Mixture Models (GMM)
% Based on initialized Priors, Mu, Sigma
% Input:
%   - Data: D x N matrix
%   - Priors: 1 x K
%   - Mu: D x K
%   - Sigma: D x D x K
% Output:
%   - Updated Priors, Mu, Sigma after EM steps

[D, N] = size(Data);
K = length(Priors);

maxIter = 100;
logLikelihood = zeros(1, maxIter);

for iter = 1:maxIter
    % E-step: compute responsibilities
    Pxi = zeros(K, N);
    for k = 1:K
        Pxi(k,:) = Priors(k) * gaussPDF(Data, Mu(:,k), Sigma(:,:,k));
    end
    Px = sum(Pxi,1) + realmin;
    gamma = Pxi ./ Px;

    % M-step: update parameters
    Nk = sum(gamma,2);
    Priors = Nk' / N;

    for k = 1:K
        Mu(:,k) = Data * gamma(k,:)' / Nk(k);
        Xc = Data - Mu(:,k);
        Sigma(:,:,k) = (Xc .* gamma(k,:)) * Xc' / Nk(k);
        Sigma(:,:,k) = Sigma(:,:,k) + 1e-6 * eye(D); % Regularization
    end

    % (Optional) Check log-likelihood
    logLikelihood(iter) = sum(log(Px));
    if iter > 2 && abs(logLikelihood(iter) - logLikelihood(iter-1)) < 1e-4
        break;
    end
end
end
