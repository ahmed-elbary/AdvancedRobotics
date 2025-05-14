addpath(genpath('./SEDS_lib')); % Add SEDS helper functions

%% STEP 1: Load demonstration data
load("D:\University of Lincoln\Semester B\Advanced Robotics\AdvancedRobotics\Assessment2\SEDS\models\recorded_motions\WShape.mat");

% Check if 'dt' is defined (required for velocity calculation)
if ~exist('dt', 'var')
    error('Time step "dt" not found in WShape.mat');
end

% Concatenate all demonstrations into one long trajectory
Xi = []; Xi_dot = [];
for i = 1:length(demos)
    demo = demos{i};     % [2 x T]
    Xi = [Xi, demo];     % Positions
    % Compute velocity using finite difference
    vel = diff(demo,1,2) / dt;
    vel(:,end+1) = vel(:,end); % Pad to keep same length
    Xi_dot = [Xi_dot, vel];    % Velocities
end

% Combine positions and velocities into one matrix: Data = [positions; velocities]
Data = [Xi; Xi_dot];

nbData = size(Data, 2);

%% STEP 2: Loop over number of Gaussians and compute BIC
nbGaussList = 2:10;
BIC_scores = zeros(1, length(nbGaussList));
models = cell(1, length(nbGaussList));

for i = 1:length(nbGaussList)
    nbStates = nbGaussList(i);
    
    % Initialize GMM parameters
    [Priors0, Mu0, Sigma0] = initialize_SEDS(Data, nbStates);
    
    % Set solver options
    opt.display = 'off';
    opt.tol_mat_bias = 1e-8;
    
    % Train SEDS model
    [Priors, Mu, Sigma] = SEDS_Solver(Data, Priors0, Mu0, Sigma0, opt);
    
    % Store model
    model.Priors = Priors;
    model.Mu = Mu;
    model.Sigma = Sigma;
    model.nbStates = nbStates;
    models{i} = model;
    
    % Compute log-likelihood and BIC
    loglik = compute_log_likelihood(Data, model);
    k = nbStates * (2*4 + 10); % Approximate number of parameters
    BIC_scores(i) = -2 * loglik + k * log(nbData);
end

%% STEP 3: Select model with best (lowest) BIC
[~, bestIdx] = min(BIC_scores);
bestModel = models{bestIdx};
best_nGauss = nbGaussList(bestIdx);
fprintf('Optimal number of Gaussians (by BIC): %d\n', best_nGauss);

%% STEP 4: Plot the results
figure('Name', 'SEDS - Learned Velocity Field');
plotSEDSStreamLines(bestModel, Xi, [1:2], [3:4], Xi(:,1), 0.01); % Streamline plotting
title(['WShape Reproduction using SEDS (n\_Gauss = ' num2str(best_nGauss) ')']);
xlabel('x'); ylabel('y');
axis equal;
grid on;
