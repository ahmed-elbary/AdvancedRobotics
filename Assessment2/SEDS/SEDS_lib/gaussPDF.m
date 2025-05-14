function prob = gaussPDF(Data, Mu, Sigma)
% GAUSSPDF computes the Probability Density Function (PDF) of a
% multivariate normal distribution.
%
% Inputs -----------------------------------------------------------------
%   o Data:  D x N matrix representing N datapoints of D dimensions.
%   o Mu:    D x 1 mean vector.
%   o Sigma: D x D covariance matrix.
%
% Outputs ----------------------------------------------------------------
%   o prob:  1 x N row vector representing the probability for each data point.

[D, N] = size(Data);
Data = Data - repmat(Mu, 1, N);  % Center the data

% Compute PDF
denom = sqrt((2*pi)^D * (abs(det(Sigma)) + realmin));
invSigma = inv(Sigma + 1e-8 * eye(D));  % Regularized inverse
expo = sum((Data' * invSigma) .* Data', 2);
prob = exp(-0.5 * expo)' / denom;
