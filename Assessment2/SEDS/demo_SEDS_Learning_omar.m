% Enhanced MATLAB script for SEDS learning with BIC-based Gaussian number selection
% and RMSE comparison
clc;
clear;
close all;

%% User Parameters and Setting
load('models/recorded_motions/Line.mat','demos')

% Pre-processing
dt = 0.1; % Time step of the demonstrations
tol_cutting = 1; % Threshold on velocity for trimming demos

% Training parameters
K_range = 1:10; % Range of Gaussian components to test
bic_values = zeros(size(K_range)); % Store BIC values
rmse_values = zeros(size(K_range)); % Store RMSE values
best_models = cell(size(K_range)); % Store model parameters

% Solver options
options.tol_mat_bias = 10^-6;
options.display = 1;
options.tol_stopping = 10^-10;
options.max_iter = 200;
options.objective = 'mse';

%% Add Libraries to Path
if isempty(regexp(path,['SEDS_lib' pathsep], 'once'))
    addpath([pwd, '/SEDS_lib']);
end
if isempty(regexp(path,['GMR_lib' pathsep], 'once'))
    addpath([pwd, '/GMR_lib']);
end

%% Preprocess Demonstrations
[x0, xT, Data, index] = preprocess_demos(demos, dt, tol_cutting);
d = size(Data,1)/2; % Data dimension
x0_all = Data(1:d, index(1:end-1)); % Initial points

%% BIC-based Model Selection and Training
for i = 1:length(K_range)
    K = K_range(i);
    fprintf('Training model with %d Gaussians...\n', K);
    
    % Initialize GMM
    [Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data, K);
    
    % Run SEDS optimization
    [Priors, Mu, Sigma] = SEDS_Solver(Priors_0, Mu_0, Sigma_0, Data, options);
    
    % Store model
    best_models{i} = struct('Priors', Priors, 'Mu', Mu, 'Sigma', Sigma);
    
    % Compute BIC
    n_params = K * (1 + 2*d + d*(d+1)/2); % Priors + Mu + Sigma
    n_data = size(Data, 2);
    fn_handle = @(x) GMR(Priors, Mu, Sigma, x, 1:d, d+1:2*d);
    xd_pred = fn_handle(Data(1:d,:));
    residuals = Data(d+1:2*d,:) - xd_pred;
    mse = mean(sum(residuals.^2, 1));
    bic_values(i) = n_data * log(mse) + n_params * log(n_data);
    
    % Compute RMSE
    rmse_values(i) = sqrt(mse);
end

% Find best model
[~, best_idx] = min(bic_values);
best_K = K_range(best_idx);
best_model = best_models{best_idx};

%% Simulate with Best Model
opt_sim.dt = 0.1;
opt_sim.i_max = 1000;
opt_sim.tol = 0.1;
fn_handle = @(x) GMR(best_model.Priors, best_model.Mu, best_model.Sigma, x, 1:d, d+1:2*d);
[x, xd] = Simulation(x0_all, [], fn_handle, opt_sim);

%% Plotting Results
% BIC and RMSE Comparison
figure('name', 'Model Selection', 'position', [100, 100, 800, 400])
subplot(1,2,1)
plot(K_range, bic_values, '-o', 'LineWidth', 2)
xlabel('Number of Gaussians (K)')
ylabel('BIC')
title('BIC vs Number of Gaussians')
grid on
subplot(1,2,2)
plot(K_range, rmse_values, '-o', 'LineWidth', 2)
xlabel('Number of Gaussians (K)')
ylabel('RMSE')
title('RMSE vs Number of Gaussians')
grid on

% Simulation Results
figure('name', 'Best Model Simulation', 'position', [265, 200, 520, 720])
sp(1) = subplot(3,1,1);
hold on; box on
plotGMM(best_model.Mu(1:2,:), best_model.Sigma(1:2,1:2,:), [0.6 1.0 0.6], 1, [0.6 1.0 0.6]);
plot(Data(1,:), Data(2,:), 'r.')
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
title(sprintf('Simulation Results (K=%d)', best_K))

sp(2) = subplot(3,1,2);
hold on; box on
plot(Data(1,:), Data(3,:), 'r.')
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\dot{\xi}_1 (mm/s)$', 'interpreter', 'latex', 'fontsize', 15);

sp(3) = subplot(3,1,3);
hold on; box on
plot(Data(2,:), Data(4,:), 'r.')
xlabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\dot{\xi}_2 (mm/s)$', 'interpreter', 'latex', 'fontsize', 15);

for i = 1:size(x,3)
    plot(sp(1), x(1,:,i), x(2,:,i), 'linewidth', 2)
    plot(sp(2), x(1,:,i), xd(1,:,i), 'linewidth', 2)
    plot(sp(3), x(2,:,i), xd(2,:,i), 'linewidth', 2)
    plot(sp(1), x(1,1,i), x(2,1,i), 'ok', 'markersize', 5, 'linewidth', 5)
    plot(sp(2), x(1,1,i), xd(1,1,i), 'ok', 'markersize', 5, 'linewidth', 5)
    plot(sp(3), x(2,1,i), xd(2,1,i), 'ok', 'markersize', 5, 'linewidth', 5)
end

for i = 1:3
    axis(sp(i), 'tight')
    ax = get(sp(i));
    axis(sp(i), [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10, ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10, ...
                 ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10, ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
    plot(sp(i), 0, 0, 'k*', 'markersize', 15, 'linewidth', 3)
end

% Streamlines
figure('name', 'Best Model Streamlines', 'position', [800, 90, 560, 320])
plotStreamLines(best_model.Priors, best_model.Mu, best_model.Sigma, [-300, 300, -300, 300])
hold on
plot(Data(1,:), Data(2,:), 'r.')
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 3)
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
title(sprintf('Streamlines (K=%d)', best_K))
set(gca, 'position', [0.1300, 0.1444, 0.7750, 0.7619])

%% Print Results
fprintf('Best number of Gaussians: %d\n', best_K);
fprintf('BIC Values:\n');
for i = 1:length(K_range)
    fprintf('K=%d: %f\n', K_range(i), bic_values(i));
end
fprintf('RMSE Values:\n');
for i = 1:length(K_range)
    fprintf('K=%d: %f\n', K_range(i), rmse_values(i));
end
%% Comparison Plot: Demonstrations vs. SEDS Simulation
figure('name', 'Demonstrations vs SEDS Simulation', 'position', [300, 150, 1000, 400])

% Plot Original Demonstrations
subplot(1,2,1)
hold on; box on
for i = 1:length(demos)
    plot(demos{i}(1,:), demos{i}(2,:), 'r', 'LineWidth', 1.5);
end
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 2)
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 14);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 14);
title('Original Demonstrations')
axis equal
grid on

% Plot SEDS Simulation
subplot(1,2,2)
hold on; box on
for i = 1:size(x,3)
    plot(x(1,:,i), x(2,:,i), 'b', 'LineWidth', 2)
end
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 2)
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 14);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 14);
title(sprintf('SEDS Simulation (K = %d)', best_K))
axis equal
grid on
