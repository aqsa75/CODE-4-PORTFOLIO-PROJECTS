% Loading the raw data
data = readtable('/Users/aqsa/Desktop/ML Project/first_ML.csv', 'VariableNamingRule', 'preserve');

% Splitting into features (X) and target variable (Y)
X = data{:, 2:end-1}; % Excluding ID and DEFAULT columns
Y = data{:, end};     % DEFAULT column is the target

% Split into training and test sets (70:30 split)
cv = cvpartition(Y, 'Holdout', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

% Standardizing the training and test data
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

% Logistic Regression Model
lr_model = fitclinear(X_train, Y_train, 'Learner', 'logistic');

% Random Forest Model
rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100);

% Predictions for both models
pred_lr = predict(lr_model, X_test);
pred_rf = predict(rf_model, X_test);

% Confusion matrix for both models
confMat_lr = confusionmat(Y_test, pred_lr);
confMat_rf = confusionmat(Y_test, pred_rf);

% Calculating Precision, Recall, F1-score for Logistic Regression
TP_lr = confMat_lr(2, 2);
FP_lr = confMat_lr(1, 2);
FN_lr = confMat_lr(2, 1);
TN_lr = confMat_lr(1, 1);

precision_lr = TP_lr / (TP_lr + FP_lr);
recall_lr = TP_lr / (TP_lr + FN_lr);
f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr);

% Calculating Precision, Recall, F1-score for Random Forest
TP_rf = confMat_rf(2, 2);
FP_rf = confMat_rf(1, 2);
FN_rf = confMat_rf(2, 1);
TN_rf = confMat_rf(1, 1);

%inspired by the github code for this from: https://github.com/preethamam/MultiClassMetrics-ConfusionMatrix/blob/main/multiclass_metrics_common.m
% ^ this repository was linked out from the mathworks website for others to
% learn how to use the function

precision_rf = TP_rf / (TP_rf + FP_rf);
recall_rf = TP_rf / (TP_rf + FN_rf);
f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf);

% Calculating AUC
[~, score_lr] = predict(lr_model, X_test);
[~, score_rf] = predict(rf_model, X_test);
[~, ~, ~, AUC_lr] = perfcurve(Y_test, score_lr(:,2), 1);
[~, ~, ~, AUC_rf] = perfcurve(Y_test, score_rf(:,2), 1);

%reference: https://stackoverflow.com/questions/23696609/what-are-the-matlab-perfcurve-roc-curve-parameters
%reference: https://uk.mathworks.com/help/stats/perfcurve.html?#bunsogv-scores

% Calculating Log Loss (Cross-Entropy Loss)

% Add small epsilon to avoid log(0) or log(1) for log loss calculation
epsilon = 1e-15;
score_rf = max(min(score_rf, 1 - epsilon), epsilon);  % Clipping the values between epsilon and 1-epsilon

log_loss_lr = logloss(Y_test, score_lr(:,2));
log_loss_rf = logloss(Y_test, score_rf(:,2));
% I kept getting the 'loss' matlab function as an option when i googled
% things but stumbled upon this github that used the 'logloss' function to
% clip things for a binary classification
% (https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/logLoss.m).
% With the help of GenAI i was able to use this function correction to
% ensure i did not make any mistakes


% Display Test Scores for both models
fprintf('Logistic Regression Test Scores:\n');
fprintf('Accuracy: %.2f\n', sum(pred_lr == Y_test) / length(Y_test));
fprintf('Recall: %.2f\n', recall_lr);
fprintf('Precision: %.2f\n', precision_lr);
fprintf('F1-score: %.2f\n', f1_lr);
fprintf('AUC: %.2f\n', AUC_lr);
fprintf('Log Loss: %.2f\n', log_loss_lr);

fprintf('\nRandom Forest Test Scores:\n');
fprintf('Accuracy: %.2f\n', sum(pred_rf == Y_test) / length(Y_test));
fprintf('Recall: %.2f\n', recall_rf);
fprintf('Precision: %.2f\n', precision_rf);
fprintf('F1-score: %.2f\n', f1_rf);
fprintf('AUC: %.2f\n', AUC_rf);
fprintf('Log Loss: %.2f\n', log_loss_rf);

% ROC Curve Plotting for Logistic Regression
figure;
[X_lr, Y_lr, ~, AUC_lr] = perfcurve(Y_test, score_lr(:,2), 1); % Getting FPR and TPR for Logistic Regression
plot(X_lr, Y_lr, 'c', 'DisplayName', sprintf('Logistic Regression (AUC = %.2f)', AUC_lr));
hold on;

% ROC Curve Plotting for Random Forest
[X_rf, Y_rf, ~, AUC_rf] = perfcurve(Y_test, score_rf(:,2), 1); % Getting FPR and TPR for Random Forest
plot(X_rf, Y_rf, 'm', 'DisplayName', sprintf('Random Forest (AUC = %.2f)', AUC_rf));

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves');
legend show;


% Plot Confusion Matrix for both models
figure;
subplot(1, 2, 1);
heatmap(confMat_lr, 'Title', 'Confusion Matrix (Logistic Regression)', 'XLabel', 'Predicted', 'YLabel', 'Actual');
subplot(1, 2, 2);
heatmap(confMat_rf, 'Title', 'Confusion Matrix (Random Forest)', 'XLabel', 'Predicted', 'YLabel', 'Actual');

% Precision vs Recall Curve for both models
figure;
plot([0, 1], [recall_lr, precision_lr], 'r', 'DisplayName', 'Logistic Regression');
hold on;
plot([0, 1], [recall_rf, precision_rf], 'b', 'DisplayName', 'Random Forest');
xlabel('Recall');
ylabel('Precision');
title('Precision vs Recall');
legend show;

% Metrics for Logistic Regression and Random Forest
metrics = {'Accuracy', 'Recall', 'Precision', 'F1-score', 'AUC', 'Log Loss'};  % List of metrics
values_lr = [sum(pred_lr == Y_test) / length(Y_test), recall_lr, precision_lr, f1_lr, AUC_lr, log_loss_lr];  % Logistic Regression values
values_rf = [sum(pred_rf == Y_test) / length(Y_test), recall_rf, precision_rf, f1_rf, AUC_rf, log_loss_rf];  % Random Forest values

% Grouped bar chart to compare metrics between Logistic Regression and Random Forest
figure;

% Creating a grouped bar chart
bar_data = [values_lr; values_rf]';  % Arrange data so that each row corresponds to a model and each column to a metric
b = bar(bar_data, 'grouped');  % Grouped bar chart

% Customizing the  bar colors so that there is one color for each metric
b(1).FaceColor = '#D8BFD8';  % blue for Logistic Regression
b(2).FaceColor = '#ADD8E6';  % pink for Random Forest

% Setting the x-axis labels (metrics)
set(gca, 'XTickLabel', metrics);

% Labeling the axes and the title
ylabel('Metric Value');
xlabel('Metrics');
title('Comparison of Logistic Regression and Random Forest across Different Metrics');

% Displaying a legend
legend({'Logistic Regression', 'Random Forest'}, 'Location', 'NorthEast');

% Adding values on top of the bars for better visibility
for i = 1:length(metrics)
    text(i-0.15, bar_data(i, 1) + 0.02, sprintf('%.2f', bar_data(i, 1)), 'FontSize', 12);  % Logistic Regression values
    text(i+0.15, bar_data(i, 2) + 0.02, sprintf('%.2f', bar_data(i, 2)), 'FontSize', 12);  % Random Forest values
end



%Defining Log Loss function with the clipped values from above
function ll = logloss(y_true, y_pred)
  ll = -mean(y_true .* log(y_pred) + (1 - y_true) .* log(1 - y_pred));
end




