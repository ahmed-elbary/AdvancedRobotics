function run_SEDS_for_shape(shapeName)
% Example usage: run_SEDS_for_shape('WShape')

clc; close all;

%% Setup
dt = 0.1;
tol_cutting = 1;
K_range = 2:10;

% Solver options
options.tol_mat_bias = 1e-6;
options.display = 0;
options.tol_stopping = 1e-10;
options.max_iter = 200;
options.objective = 'mse';

% Add required libraries
addpath(genpath('./SEDS_lib'));
addpath(genpath('./GMR_lib'));

%% Load and preprocess demonstrations
matPath = fullfile('models', 'recorded_motions', [shapeName '.mat']);
load(matPath, 'demos');
[x0, xT, Data, index] = preprocess_demos(demos, dt, tol_cutting);
d = size(Data,1)/2;
demo = demos{1};  % Use first demonstration

%% BIC Evaluation Loop
bic_values = zeros(size(K_range));
models = cell(size(K_range));

for i = 1:length(K_range)
    K = K_range(i);
    [Priors0, Mu0, Sigma0] = initialize_SEDS(Data, K);
    [Priors, Mu, Sigma] = SEDS_Solver(Priors0, Mu0, Sigma0, Data, options);

    models{i} = struct('Priors', Priors, 'Mu', Mu, 'Sigma', Sigma, 'K', K);

    % BIC computation
    xd_pred = GMR(Priors, Mu, Sigma, Data(1:d,:), 1:d, d+1:2*d);
    residuals = Data(d+1:2*d,:) - xd_pred;
    mse = mean(sum(residuals.^2, 1));

    n_params = K * (1 + 2*d + d*(d+1)/2);
    n_data = size(Data,2);
    bic_values(i) = n_data * log(mse) + n_params * log(n_data);
end

% Select best model
[~, best_idx] = min(bic_values);
best_model = models{best_idx};
best_K = K_range(best_idx);
fprintf('%s: Best K = %d\n', shapeName, best_K);

%% Plot 1: BIC vs K
figure;
plot(K_range, bic_values, '-o', 'LineWidth', 2);
xlabel('Number of Gaussians (K)');
ylabel('BIC');
title([shapeName ' - BIC vs Number of Gaussians']);
grid on;
saveas(gcf, [shapeName '_SEDS_BIC_plot.png']);

%% Plot 2: Best Trajectory
% Simulate from first demo start
opt_sim.dt = dt;
opt_sim.i_max = 1000;
opt_sim.tol = 0.1;
fn = @(x) GMR(best_model.Priors, best_model.Mu, best_model.Sigma, x, 1:d, d+1:2*d);
[x_sim, ~] = Simulation(demo(:,1), [], fn, opt_sim);

% Plot
figure;
hold on;
plot(demo(1,:), demo(2,:), 'k--', 'LineWidth', 2);             % Demo
plot(x_sim(1,:), x_sim(2,:), 'b-', 'LineWidth', 2);            % Prediction
plot(demo(1,1), demo(2,1), 'go', 'MarkerSize', 10, 'LineWidth', 2); % Start
plot(0, 0, 'r*', 'MarkerSize', 12, 'LineWidth', 2);            % Goal
legend('Demonstration', 'SEDS Prediction', 'Start', 'Goal')
xlabel('\xi_1'); ylabel('\xi_2');
title(sprintf('%s - Best SEDS Trajectory (K = %d)', shapeName, best_K));
axis equal;
grid on;
saveas(gcf, [shapeName '_SEDS_CleanTrajectory.png']);
end
