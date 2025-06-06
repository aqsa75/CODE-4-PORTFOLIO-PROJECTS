% Loading the raw data
data = readtable('/Users/aqsa/Desktop/ML Project/first_ML.csv', 'VariableNamingRule', 'preserve');

% Splitting into features (X) and target variable (Y)
X = data{:, 2:end-1}; % Excluding ID and DEFAULT columns
Y = data{:, end};     % DEFAULT column is the target

% Split into training and test sets (70:30 split)
cv = cvpartition(Y, 'Holdout', 0.3);  % 30% test set
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));


% Standardize the training and test data (code inspired from:
% https://gist.github.com/jsouza/4d2a8a3ba47bc82075d9)
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

fprintf('Starting Logistic Regression Tuning...\n'); %had to look up this functionality (https://uk.mathworks.com/help/matlab/ref/tic.html)
tic; % Starts the  timing

% Random Search for Logistic Regression
lambda_random = logspace(-4, 0, 10); % Random search over regularization
solver_random = {'lbfgs', 'sgd', 'sparsa'}; % Random solvers for optimization
best_mcr_lr = inf;

for i = 1:length(lambda_random) %GENAI helped me debug my initial loop 
    for j = 1:length(solver_random)
        % K-fold cross-validation setup
        cv_lr = cvpartition(Y_train, 'KFold', 5);
        mcr_fold = zeros(cv_lr.NumTestSets, 1);

        for k = 1:cv_lr.NumTestSets
            % Train-validation split
            X_train_fold = X_train(training(cv_lr, k), :);
            Y_train_fold = Y_train(training(cv_lr, k));
            X_val_fold = X_train(test(cv_lr, k), :);
            Y_val_fold = Y_train(test(cv_lr, k));

            % Train logistic regression
            model_lr = fitclinear(X_train_fold, Y_train_fold, 'Learner', 'logistic', ...
                'Lambda', lambda_random(i), 'Solver', solver_random{j});

            % Predict and calculate misclassification rate
            pred_val = predict(model_lr, X_val_fold);
            mcr_fold(k) = mean(pred_val ~= Y_val_fold);
        end

        % Evaluate the average misclassification rate
        avg_mcr = mean(mcr_fold);
        if avg_mcr < best_mcr_lr
            best_mcr_lr = avg_mcr;
            best_lambda_lr = lambda_random(i);
            best_solver_lr = solver_random{j};
        end
    end
end

% Train final logistic regression model with best hyperparameters
final_lr_model = fitclinear(X_train, Y_train, 'Learner', 'logistic', ...
    'Lambda', best_lambda_lr, 'Solver', best_solver_lr);

time_lr = toc; % End timing
fprintf('Best Logistic Regression - Lambda: %.4f, Solver: %s, MCR: %.4f\n', ...
    best_lambda_lr, best_solver_lr, best_mcr_lr);
fprintf('Logistic Regression Computational Time: %.2f seconds\n', time_lr);


%% Random Forest: Cross-Validation and Hyperparameter Tuning
fprintf('Starting Random Forest Tuning...\n');
tic; % Start timing

% Random Search: Define broad hyperparameter space
num_trees_random = [50, 100, 150];       % Number of trees
min_leaf_sizes_random = [1, 5, 10];      % Minimum leaf size (applies to trees via template)
best_auc_rf = -inf;                      % Initialize best AUC

% Random Search
%GENAI helped me debug my initial loop 
for i = 1:length(num_trees_random)
    for j = 1:length(min_leaf_sizes_random)
        % Define tree template with current min leaf size
        tree_template = templateTree('MinLeafSize', min_leaf_sizes_random(j));
        
        % K-fold cross-validation setup
        cv_rf = cvpartition(Y_train, 'KFold', 5);
        auc_fold = zeros(cv_rf.NumTestSets, 1); % Store AUC for each fold

        for l = 1:cv_rf.NumTestSets
            % Train-validation split
            X_train_fold = X_train(training(cv_rf, l), :);
            Y_train_fold = Y_train(training(cv_rf, l));
            X_val_fold = X_train(test(cv_rf, l), :);
            Y_val_fold = Y_train(test(cv_rf, l));

            % Train Random Forest
            model_rf = fitcensemble(X_train_fold, Y_train_fold, 'Method', 'Bag', ...
                'NumLearningCycles', num_trees_random(i), 'Learners', tree_template);

            % Predict probabilities and calculate AUC
            [~, scores] = predict(model_rf, X_val_fold);
            [~, ~, ~, auc] = perfcurve(Y_val_fold, scores(:, 2), 1);
            auc_fold(l) = auc;
        end

        % Average AUC across folds
        avg_auc = mean(auc_fold);

        % Update best parameters if AUC improves
        if avg_auc > best_auc_rf
            best_auc_rf = avg_auc;
            best_num_trees_rf = num_trees_random(i);
            best_min_leaf_size_rf = min_leaf_sizes_random(j);
        end
    end
end

% this is a more focused Grid Search around the best values found in my Random Search
fprintf('Starting Random Forest Grid Search...\n');
num_trees_grid = best_num_trees_rf - 20:10:best_num_trees_rf + 20; % Refine around best number of trees
min_leaf_sizes_grid = max(1, best_min_leaf_size_rf - 2):best_min_leaf_size_rf + 2; % Refine leaf sizes
best_auc_rf_grid = -inf;

%GENAI helped me debug my this loop  below 
for i = 1:length(num_trees_grid)
    for j = 1:length(min_leaf_sizes_grid)
        % Defined the tree template with current min leaf size
        tree_template = templateTree('MinLeafSize', min_leaf_sizes_grid(j));
        %referenced this for the code above: https://uk.mathworks.com/help/stats/templatetree.html

        % K-fold cross-validation setup
        cv_rf = cvpartition(Y_train, 'KFold', 5);
        auc_fold = zeros(cv_rf.NumTestSets, 1);

        for l = 1:cv_rf.NumTestSets
            % Train-validation split
            X_train_fold = X_train(training(cv_rf, l), :);
            Y_train_fold = Y_train(training(cv_rf, l));
            X_val_fold = X_train(test(cv_rf, l), :);
            Y_val_fold = Y_train(test(cv_rf, l));

            % Train Random Forest
            model_rf = fitcensemble(X_train_fold, Y_train_fold, 'Method', 'Bag', ...
                'NumLearningCycles', num_trees_grid(i), 'Learners', tree_template);

            % Predict probabilities and calculate AUC
            [~, scores] = predict(model_rf, X_val_fold);
            [~, ~, ~, auc] = perfcurve(Y_val_fold, scores(:, 2), 1);
            auc_fold(l) = auc;
        end

        % Averaged AUC across folds
        avg_auc = mean(auc_fold);

        % Updated best parameters if AUC improves
        if avg_auc > best_auc_rf_grid
            best_auc_rf_grid = avg_auc;
            best_num_trees_rf_grid = num_trees_grid(i);
            best_min_leaf_size_rf_grid = min_leaf_sizes_grid(j);
        end
    end
end

% Train final Random Forest model using the best hyperparameters
final_tree_template = templateTree('MinLeafSize', best_min_leaf_size_rf_grid);
final_rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', ...
    'NumLearningCycles', best_num_trees_rf_grid, 'Learners', final_tree_template);

time_rf = toc; % End timing

% Display Results
fprintf('Best Random Forest (Grid Search) - NumTrees: %d, MinLeafSize: %d, AUC: %.4f\n', ...
    best_num_trees_rf_grid, best_min_leaf_size_rf_grid, best_auc_rf_grid);
fprintf('Random Forest Computational Time: %.2f seconds\n', time_rf);

%% Final Model Evaluation on Test Set
% Random Forest
[pred_rf, score_rf] = predict(final_rf_model, X_test);
[~, ~, ~, AUC_rf] = perfcurve(Y_test, score_rf(:, 2), 1);

% Display Final Test Metrics
fprintf('\nFinal Random Forest AUC: %.4f\n', AUC_rf);

%% Final Model Evaluation on Test Set
% Logistic Regression
[pred_lr, score_lr] = predict(final_lr_model, X_test);
[~, ~, ~, AUC_lr] = perfcurve(Y_test, score_lr(:, 2), 1);

% Random Forest
[pred_rf, score_rf] = predict(final_rf_model, X_test);
[~, ~, ~, AUC_rf] = perfcurve(Y_test, score_rf(:, 2), 1);

% Display Final Test Metrics
fprintf('\nFinal Logistic Regression AUC: %.4f\n', AUC_lr);
fprintf('Final Random Forest AUC: %.4f\n', AUC_rf);

%% Confusion Matrix and Visualization
% Logistic Regression Confusion Matrix
cm_lr = confusionmat(Y_test, pred_lr);
figure;
confusionchart(cm_lr, unique(Y_test), 'Title', 'Logistic Regression Confusion Matrix', ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

% Random Forest Confusion Matrix
cm_rf = confusionmat(Y_test, pred_rf);
figure;
confusionchart(cm_rf, unique(Y_test), 'Title', 'Random Forest Confusion Matrix', ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');

%% Summary
fprintf('\nConfusion matrices plotted for both models.\n');
