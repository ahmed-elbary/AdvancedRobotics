% Enhanced MATLAB script for SEDS learning with hyperparameter tuning
clc;
clear;
close all;

%% User Parameters and Setting
load('models/recorded_motions/Line.mat','demos')

% Pre-processing
dt = 0.1; % Time step of the demonstrations
tol_cutting = 1; % Threshold on velocity for trimming demos

%% Hyperparameter Ranges to Tune
K_range = 1:10; % Number of Gaussian components
objective_list = {'mse', 'likelihood'}; % Optimization objectives
tol_stopping_vals = [1e-6, 1e-8]; % Stopping tolerances
max_iter_vals = [200, 400]; % Maximum iterations

% Initialize results storage
results = struct('K', {}, 'objective', {}, 'tol_stopping', {}, ...
                'max_iter', {}, 'bic', {}, 'rmse', {}, 'model', {});

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

%% Hyperparameter Tuning Loop
counter = 1;
total_combinations = length(K_range)*length(objective_list)*length(tol_stopping_vals)*length(max_iter_vals);
fprintf('Total parameter combinations to test: %d\n', total_combinations);

for K = K_range
    for obj_idx = 1:length(objective_list)
        for tol_idx = 1:length(tol_stopping_vals)
            for iter_idx = 1:length(max_iter_vals)
                % Set current options
                options.tol_mat_bias = 10^-6;
                options.display = 0; % Turn off display during tuning
                options.tol_stopping = tol_stopping_vals(tol_idx);
                options.max_iter = max_iter_vals(iter_idx);
                options.objective = objective_list{obj_idx};
                
                fprintf('Testing K=%d, obj=%s, tol=%.0e, max_iter=%d...\n', ...
                        K, options.objective, options.tol_stopping, options.max_iter);
                
                % Initialize GMM
                [Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data, K);
                
                % Run SEDS optimization
                [Priors, Mu, Sigma] = SEDS_Solver(Priors_0, Mu_0, Sigma_0, Data, options);
                
                % Compute BIC
                n_params = K * (1 + 2*d + d*(d+1)/2); % Priors + Mu + Sigma
                n_data = size(Data, 2);
                fn_handle = @(x) GMR(Priors, Mu, Sigma, x, 1:d, d+1:2*d);
                xd_pred = fn_handle(Data(1:d,:));
                residuals = Data(d+1:2*d,:) - xd_pred;
                mse = mean(sum(residuals.^2, 1));
                bic = n_data * log(mse) + n_params * log(n_data);
                
                % Compute RMSE
                rmse = sqrt(mse);
                
                % Store results
                results(counter).K = K;
                results(counter).objective = options.objective;
                results(counter).tol_stopping = options.tol_stopping;
                results(counter).max_iter = options.max_iter;
                results(counter).bic = bic;
                results(counter).rmse = rmse;
                results(counter).model = struct('Priors', Priors, 'Mu', Mu, 'Sigma', Sigma);
                
                counter = counter + 1;
            end
        end
    end
end

%% Find Best Model Based on BIC
[~, best_idx] = min([results.bic]);
best_result = results(best_idx);

fprintf('\nBest model parameters:\n');
fprintf('K = %d\n', best_result.K);
fprintf('Objective = %s\n', best_result.objective);
fprintf('Tolerance = %.1e\n', best_result.tol_stopping);
fprintf('Max Iterations = %d\n', best_result.max_iter);
fprintf('BIC = %.4f | RMSE = %.4f\n', best_result.bic, best_result.rmse);

%% Simulation with Best Model
opt_sim.dt = 0.1;
opt_sim.i_max = 1000;
opt_sim.tol = 0.1;
fn_handle = @(x) GMR(best_result.model.Priors, best_result.model.Mu, ...
                     best_result.model.Sigma, x, 1:d, d+1:2*d);
[x, xd] = Simulation(x0_all, [], fn_handle, opt_sim);

%% Visualization: Demonstrations vs SEDS Trajectories
figure('name', 'Demonstrations vs SEDS Trajectories', 'position', [100, 100, 1200, 500]);

% Plot Demonstrations
subplot(1,2,1);
hold on; box on;
title('Original Demonstrations');
plot(Data(1,:), Data(2,:), 'r.', 'MarkerSize', 10);
for i = 1:length(demos)
    plot(demos{i}(1,:), demos{i}(2,:), 'b-', 'LineWidth', 1.5);
end
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 3);
axis tight;
grid on;

% Plot SEDS Generated Trajectories
subplot(1,2,2);
hold on; box on;
title(sprintf('SEDS Generated (K=%d, %s)', best_result.K, best_result.objective));
plot(Data(1,:), Data(2,:), 'r.', 'MarkerSize', 10);
plotGMM(best_result.model.Mu(1:2,:), best_result.model.Sigma(1:2,1:2,:), ...
       [0.6 1.0 0.6], 1, [0.6 1.0 0.6]);
for i = 1:size(x,3)
    plot(x(1,:,i), x(2,:,i), 'b-', 'linewidth', 1.5);
    plot(x(1,1,i), x(2,1,i), 'ko', 'markersize', 5, 'linewidth', 2);
end
xlabel('$\xi_1 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
ylabel('$\xi_2 (mm)$', 'interpreter', 'latex', 'fontsize', 15);
plot(0, 0, 'k*', 'markersize', 15, 'linewidth', 3);
axis tight;
grid on;

%% Performance Analysis Plots
% BIC vs K for different objectives
figure('name', 'BIC Analysis', 'position', [100, 100, 800, 400]);
hold on;
colors = lines(length(objective_list));
for obj_idx = 1:length(objective_list)
    mask = strcmp({results.objective}, objective_list{obj_idx});
    [sorted_K, sort_idx] = sort([results(mask).K]);
    bic_values = [results(mask).bic];
    bic_values = bic_values(sort_idx);
    plot(sorted_K, bic_values, '-o', 'Color', colors(obj_idx,:), ...
         'LineWidth', 2, 'DisplayName', objective_list{obj_idx});
end
xlabel('Number of Gaussians (K)');
ylabel('BIC');
title('BIC vs Number of Gaussians for Different Objectives');
legend('show');
grid on;

% RMSE vs K for different objectives
figure('name', 'RMSE Analysis', 'position', [100, 100, 800, 400]);
hold on;
for obj_idx = 1:length(objective_list)
    mask = strcmp({results.objective}, objective_list{obj_idx});
    [sorted_K, sort_idx] = sort([results(mask).K]);
    rmse_values = [results(mask).rmse];
    rmse_values = rmse_values(sort_idx);
    plot(sorted_K, rmse_values, '-o', 'Color', colors(obj_idx,:), ...
         'LineWidth', 2, 'DisplayName', objective_list{obj_idx});
end
xlabel('Number of Gaussians (K)');
ylabel('RMSE');
title('RMSE vs Number of Gaussians for Different Objectives');
legend('show');
grid on;

%% Display All Results in a Table
fprintf('\nAll Results Summary:\n');
fprintf('%-5s %-12s %-12s %-10s %-12s %-12s\n', ...
        'K', 'Objective', 'Tolerance', 'MaxIter', 'BIC', 'RMSE');
for i = 1:length(results)
    fprintf('%-5d %-12s %-12.1e %-10d %-12.4f %-12.4f\n', ...
            results(i).K, results(i).objective, results(i).tol_stopping, ...
            results(i).max_iter, results(i).bic, results(i).rmse);
end