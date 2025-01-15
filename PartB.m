data = readtable("A1081 June.csv", "NumHeaderLines", 3);

training_data = data(data.LocalDate == "2016-06-02" | data.LocalDate == "2016-06-09" | data.LocalDate == "2016-06-16" | data.LocalDate == "2016-06-23", :);
test_data = data(data.LocalDate == "2016-06-30", :);

X_test = minutes(test_data.LocalTime);
y_test = test_data.TotalCarriagewayFlow;
X_train = minutes(training_data.LocalTime);
y_train = training_data.TotalCarriagewayFlow;

kernel_types = {'linear', 'gaussian', 'polynomial'};

linear_params = {
    {'BoxConstraint', 10}, 
    {'BoxConstraint', 100}, 
    {'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus')}
};

gaussian_params = {
    {'BoxConstraint', 10, 'KernelScale', 1}, 
    {'BoxConstraint', 100, 'KernelScale', 10}, 
    {'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus')}
};

polynomial_params = {
    {'PolynomialOrder', 5, 'BoxConstraint', 1000}, 
    {'PolynomialOrder', 9, 'BoxConstraint', 1000}, 
    {'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus')}
};

results_table = table();

svr_optimized = cell(1, length(kernel_types));

kernel = 'linear';
fprintf('\nTesting kernel: %s\n', kernel);

parameters = {};
svr_optimized{1, 1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel);  % Default parameters
hyperparams = 'Default';

[results_table, svr_optimized{1, 1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{1, 1}, hyperparams, kernel, results_table);


for i = 1:2
    params = linear_params{i};
    svr_optimized{1, i+1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
    hyperparams = sprintf('%s', strjoin(cellfun(@(x) sprintf('%s: %d', x{1}, x{2}), num2cell(params, 2), 'UniformOutput', false), ', '));
    [results_table, svr_optimized{1, i+1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{1, i+1}, hyperparams, kernel, results_table);
end


params = linear_params{3};
svr_optimized{1, 3} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
hyperparams = 'Optimized';
[results_table, svr_optimized{1, 3}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{1, 3}, hyperparams, kernel, results_table);


kernel = 'gaussian';
fprintf('\nTesting kernel: %s\n', kernel);

parameters = {};
svr_optimized{2, 1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel);
hyperparams = 'Default';
[results_table, svr_optimized{2, 1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{2, 1}, hyperparams, kernel, results_table);

for i = 1:2
    params = gaussian_params{i};
    svr_optimized{2, i+1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
    hyperparams = sprintf('%s', strjoin(cellfun(@(x) sprintf('%s: %d', x{1}, x{2}), num2cell(params, 2), 'UniformOutput', false), ', '));
    [results_table, svr_optimized{2, i+1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{2, i+1}, hyperparams, kernel, results_table);
end

params = gaussian_params{3};
svr_optimized{2, 3} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
hyperparams = 'Optimized';
[results_table, svr_optimized{2, 3}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{2, 3}, hyperparams, kernel, results_table);

kernel = 'polynomial';
fprintf('\nTesting kernel: %s\n', kernel);

parameters = {};
svr_optimized{3, 1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel);
hyperparams = 'Default';
[results_table, svr_optimized{3, 1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{3, 1}, hyperparams, kernel, results_table);

for i = 1:2
    params = polynomial_params{i};
    svr_optimized{3, i+1} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
    hyperparams = sprintf('%s', strjoin(cellfun(@(x) sprintf('%s: %d', x{1}, x{2}), num2cell(params, 2), 'UniformOutput', false), ', '));
    [results_table, svr_optimized{3, i+1}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{3, i+1}, hyperparams, kernel, results_table);
end

params = polynomial_params{3};
svr_optimized{3, 3} = fitrsvm(X_train, y_train, 'KernelFunction', kernel, params{:});
hyperparams = 'Optimized';
[results_table, svr_optimized{3, 3}, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, svr_optimized{3, 3}, hyperparams, kernel, results_table);

disp(results_table);

function [results_table, model, hyperparams] = train_and_store_results(X_train, y_train, X_test, y_test, model, hyperparams, kernel, results_table)
    y_train_pred = predict(model, X_train);
    y_test_pred = predict(model, X_test);

    mse_train = mean((y_train - y_train_pred).^2);
    mse_test = mean((y_test - y_test_pred).^2);
    r2_train = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);
    r2_test = 1 - sum((y_test - y_test_pred).^2) / sum((y_test - mean(y_test)).^2);
    
    new_row = table(string(kernel), string(hyperparams), mse_train, mse_test, r2_train, r2_test);
    results_table = [results_table; new_row];
    
    fprintf('Kernel: %s\n', kernel);
    fprintf('Parameters: %s\n', hyperparams);
    fprintf('Training MSE: %.4f, Training R^2: %.4f\n', mse_train, r2_train);
    fprintf('Test MSE: %.4f, Test R^2: %.4f\n\n', mse_test, r2_test);
end

figure;
scatter(X_train, y_train, 'b.', 'DisplayName', 'Training Data'); % Scatter plot for training data
hold on;
scatter(X_test, y_test, 'r.', 'DisplayName', 'Test Data'); % Scatter plot for test data

X_smooth = linspace(min([X_train; X_test]), max([X_train; X_test]), 1000)'; % 100 points for smooth curve

y_pred_linear = predict(svr_optimized{1, 3}, X_smooth);
plot(X_smooth, y_pred_linear, 'b-', 'LineWidth', 2, 'DisplayName', 'SVM Linear Kernel Model');

y_pred_gaussian = predict(svr_optimized{2, 3}, X_smooth);
plot(X_smooth, y_pred_gaussian, 'g-', 'LineWidth', 2, 'DisplayName', 'SVM Gaussian Kernel Model');

y_pred_polynomial = predict(svr_optimized{3, 3}, X_smooth);
plot(X_smooth, y_pred_polynomial, 'k-', 'LineWidth', 2, 'DisplayName', 'SVM Polynomial Kernel Model');

poly_fit = fit(X_train, y_train, sprintf('poly%d', 9));
y_pred_poly_fit = feval(poly_fit, X_smooth);
plot(X_smooth, y_pred_poly_fit,'y-','LineWidth', 2, 'DisplayName', 'LSR Polynomial Model of Order 9')

title('SVR Models with Different Kernels: Predictions on Train and Test Data');
xlabel('Time (minutes)');
ylabel('Carriageway Flow');

legend('show');

hold off;