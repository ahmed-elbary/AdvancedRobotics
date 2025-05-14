clc; clear; close all;

%% STEP 1: Load WShape Data
load('models/recorded_motions/WShape.mat', 'demos')
dt = 0.1;           % Time step
tol_cutting = 1;    % Trimming threshold

% Add libraries
addpath(genpath('./SEDS_lib'));
addpath(genpath('./GMR_lib'));

%% STEP 2: Preprocess demonstrations
[x0, xT, Data, index] = preprocess_demos(demos, dt, tol_cutting);
d = size(Data, 1) / 2;

%% STEP 3: Train with BIC-based model selection
K_range = 2:10;
bic_values = zeros(size(K_range));
models = cell(size(K_range));

options.tol_mat_bias = 1e-6;
options.display = 0;
options.tol_stopping = 1e-10;
options.max_iter = 200;
options.objective = 'mse';

for i = 1:length(K_range)
    K = K_range(i);
    [Priors0, Mu0, Sigma0] = initialize_SEDS(Data, K);
    [Priors, Mu, Sigma] = SEDS_Solver(Priors0, Mu0, Sigma0, Data, options);
    models{i} = struct('Priors', Priors, 'Mu', Mu, 'Sigma', Sigma, 'K', K);
    
    % GMR prediction and BIC computation
    xd_pred = GMR(Priors, Mu, Sigma, Data(1:d,:), 1:d, d+1:2*d);
    residuals = Data(d+1:2*d,:) - xd_pred;
    mse = mean(sum(residuals.^2, 1));
    
    n_params = K * (1 + 2*d + d*(d+1)/2);
    n_data = size(Data, 2);
    bic_values(i) = n_data * log(mse) + n_params * log(n_data);
end

%% STEP 4: Choose best model by BIC
[~, best_idx] = min(bic_values);
best_model = models{best_idx};
best_K = K_range(best_idx);
fprintf('Best number of Gaussians (by BIC): %d\n', best_K);

%% STEP 5: Plot and Save

% Plot 1: BIC vs Number of Gaussians
figure;
plot(K_range, bic_values, '-o', 'LineWidth', 2)
xlabel('Number of Gaussians (K)')
ylabel('BIC')
title('BIC vs Number of Gaussians')
grid on
saveas(gcf, 'WShape_SEDS_BIC_plot.png')

% Plot 2: Streamlines for Best Model (trajectory reproduction)
figure;
plotStreamLines(best_model.Priors, best_model.Mu, best_model.Sigma, [-300 300 -300 300])
hold on
plot(Data(1,:), Data(2,:), 'r.')
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 2)
xlabel('\xi_1'); ylabel('\xi_2');
title(sprintf('WShape Reproduction using SEDS (K = %d)', best_K));
axis equal
grid on
saveas(gcf, 'WShape_SEDS_BestTrajectory.png')
