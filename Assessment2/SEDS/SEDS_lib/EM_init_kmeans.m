function [Priors, Mu, Sigma] = EM_init_kmeans(Data, K)
% EM_init_kmeans: Initializes GMM parameters using k-means clustering.
% Inputs:
%   - Data: A (D x N) matrix, where D is the data dimension and N is the number of samples.
%   - K: The number of Gaussian components.
% Outputs:
%   - Priors: 1 x K vector of prior probabilities.
%   - Mu: D x K matrix of means.
%   - Sigma: D x D x K array of covariance matrices.

[D, N] = size(Data);

% Use k-means to partition the data
% Data' is (N x D) because kmeans in MATLAB expects observations as rows.
[labels, mu] = kmeans(Data', K, 'MaxIter', 1000, 'Replicates', 5);

% Transpose means to get Mu: (D x K)
Mu = mu';

% Initialize Priors and Sigma
Priors = zeros(1, K);
Sigma = zeros(D, D, K);

for i = 1:K
    idx = find(labels == i);
    Priors(i) = length(idx) / N;
    
    if length(idx) > 1
        % Compute covariance from the selected samples (ensure correct shape)
        Sigma(:,:,i) = cov(Data(:, idx)') + 1e-3 * eye(D);
    else
        % If only one data point, use identity matrix
        Sigma(:,:,i) = eye(D);
    end
end

end
