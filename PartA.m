data = readtable("A1081 June.csv", "NumHeaderLines", 3)

training_data = data(data.LocalDate == "2016-06-02" | data.LocalDate == "2016-06-09" | data.LocalDate == "2016-06-16" | data.LocalDate == "2016-06-23", :);
test_data = data(data.LocalDate == "2016-06-30", :);

X_test = minutes(test_data.LocalTime);
y_test = test_data.TotalCarriagewayFlow;
X_train = minutes(training_data.LocalTime);
y_train = training_data.TotalCarriagewayFlow;

figure;
hold on;
scatter(X_train, y_train, 'b.');
scatter(X_test, y_test, 'r.');
xlabel('Time (Minutes)');
ylabel('Total Carriageway Flow');
title('Training and Test Data');
legend({'Training Data', 'Test Data'});
grid on;

colors = lines(9);
results_analytic = zeros(9, 4); 

figure;
hold on;
for n = 1:9
    X_train_poly = zeros(length(X_train), n+1);
    X_test_poly = zeros(length(X_test), n+1);
    
    for i = 0:n
        X_train_poly(:, i+1) = X_train.^i;
        X_test_poly(:, i+1) = X_test.^i;
    end
    
    beta = (X_train_poly' * X_train_poly) \ (X_train_poly' * y_train);
    y_pred_train = X_train_poly * beta;
    y_pred_test = X_test_poly * beta;
    
    mse_train = mean((y_train - y_pred_train).^2);
    mse_test = mean((y_test - y_pred_test).^2);
    r2_train = 1 - sum((y_train - y_pred_train).^2) / sum((y_train - mean(y_train)).^2);
    r2_test = 1 - sum((y_test - y_pred_test).^2) / sum((y_test - mean(y_test)).^2);
    
    results_analytic(n, :) = [mse_train, mse_test, r2_train, r2_test];

    X_fine = linspace(min(X_train), max(X_train), 500)';
    X_fine_poly = zeros(length(X_fine), n+1);
    for i = 0:n
        X_fine_poly(:, i+1) = X_fine.^i;
    end
    y_fine = X_fine_poly * beta;
    
    plot(X_fine, y_fine, 'Color', colors(n, :), 'LineWidth', 2);
end


disp('Analytic Results:');
result_table_analytic = array2table(results_analytic, ...
    'VariableNames', {'MSE_Train', 'MSE_Test', 'R2_Train', 'R2_Test'}, ...
    'RowNames', {'n=1','n=2', 'n=3', 'n=4', 'n=5', 'n=6', 'n=7', 'n=8', 'n=9'});
disp(result_table_analytic)

scatter(X_train, y_train, 'b.');
scatter(X_test, y_test, 'r.');

colors = lines(9);
results_analytic = zeros(9, 4); 

xlabel('Time (Minutes)');
ylabel('Total Carriageway Flow');
title('Polynomial Fits for Different n with Training and Test Data');
legend_entries = arrayfun(@(n) sprintf('Poly Fit (n=%d)', n), 1:9, 'UniformOutput', false);
legend([legend_entries, {'Training Data', 'Test Data'}], 'Location', 'Best');
grid on;

orders = 1:9; % 多项式阶数
mse_train = zeros(length(orders), 1);
mse_test = zeros(length(orders), 1);
r2_train = zeros(length(orders), 1);
r2_test = zeros(length(orders), 1);

for i = 1:9
    order = orders(i);
    
    poly_fit = fit(X_train, y_train, sprintf('poly%d', order));
    
    y_train_pred = feval(poly_fit, X_train);
    y_test_pred = feval(poly_fit, X_test);

    mse_train(i) = mean((y_train - y_train_pred).^2);
    mse_test(i) = mean((y_test - y_test_pred).^2);

    r2_train(i) = 1 - sum((y_train - y_train_pred).^2) / sum((y_train - mean(y_train)).^2);
    r2_test(i) = 1 - sum((y_test - y_test_pred).^2) / sum((y_test - mean(y_test)).^2);
    if order == 9
    disp('Coefficients for the 9th-order model (from fit, formatted to 6 decimal places):');

    coeffs_fit = coeffvalues(poly_fit);

end

end


result_table_fit = table(mse_train, mse_test, r2_train, r2_test, ...
    'VariableNames', {'TrainMSE', 'TestMSE', 'TrainR2', 'TestR2'}, ...
    'RowNames', {'n=1', 'n=2', 'n=3', 'n=4', 'n=5', 'n=6', 'n=7', 'n=8', 'n=9'});
disp(result_table_fit);


figure;
subplot(2, 1, 1);
plot(orders, mse_train, '-o', 'DisplayName', 'Training MSE');
hold on;
plot(orders, mse_test, '-o', 'DisplayName', 'Test MSE');
xlabel('Polynomial Order');
ylabel('Mean Squared Error');
title('Training and Test MSE vs Polynomial Order');
legend;
grid on;

subplot(2, 1, 2);
plot(orders, r2_train, '-o', 'DisplayName', 'Training R^2');
hold on;
plot(orders, r2_test, '-o', 'DisplayName', 'Test R^2');
xlabel('Polynomial Order');
ylabel('R^2 Score');
title('Training and Test R^2 vs Polynomial Order');
legend;
grid on;

disp('Coefficients for the 9th-order model (formatted to 30 decimal places):');
for i = 1:length(beta)
    fprintf('Coefficient %d: %.30f\n', i-1, beta(i)); 
end

