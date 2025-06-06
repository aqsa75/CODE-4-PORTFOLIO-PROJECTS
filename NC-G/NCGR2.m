% Load the dataset
df = readtable('/Users/aqsa/Desktop/NC-G/churndata.csv');

% Select features and target
X = df(:, {'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', ...
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', ...
    'Geography_France', 'Geography_Germany', 'Geography_Spain'});  % Features
y = df.Exited;  % Target (binary classification)

% Convert boolean columns to integers (True → 1, False → 0)
X = table2array(X);
y = table2array(y);

% Split into train-test sets (80% train, 20% test)
cv = cvpartition(height(df), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv));
y_test = y(test(cv));

% Sigmoid activation and derivative functions
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivative = @(x) x .* (1 - x);

% Function to create and train the model
function [losses, test_accuracy, training_time] = train_neural_network(X_train, y_train, X_test, y_test, learning_rate, hidden_neurons, epochs)
    input_size = size(X_train, 2);  % Number of input features
    output_size = 1;  % Output neuron (binary classification)

    % Initialize weights and biases
    W1 = randn(input_size, hidden_neurons) * 0.01;
    b1 = zeros(1, hidden_neurons);
    W2 = randn(hidden_neurons, output_size) * 0.01;
    b2 = zeros(1, output_size);

    % Initialize loss tracking
    losses = [];

    % Start timer to track training time
    tic;
    
    % Training loop
    for epoch = 1:epochs
        % Forward pass
        Z1 = X_train * W1 + repmat(b1, size(X_train, 1), 1);
        A1 = sigmoid(Z1);
        Z2 = A1 * W2 + repmat(b2, size(X_train, 1), 1);
        A2 = sigmoid(Z2);  % Output layer prediction

        % Calculate loss (Binary Cross-Entropy)
        loss = -mean(y_train .* log(A2) + (1 - y_train) .* log(1 - A2));
        losses = [losses; loss];

        % Backpropagation
        dA2 = A2 - y_train;
        dZ2 = dA2 .* sigmoid_derivative(A2);
        dW2 = A1' * dZ2;
        db2 = sum(dZ2, 1);

        dA1 = dZ2 * W2';
        dZ1 = dA1 .* sigmoid_derivative(A1);
        dW1 = X_train' * dZ1;
        db1 = sum(dZ1, 1);

        % Update weights and biases using gradient descent
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;

        % Print progress every 100 epochs
        if mod(epoch, 100) == 0
            fprintf('Epoch %d: Loss = %.4f\n', epoch, loss);
        end
    end
    
    % Calculate the time taken for training
    training_time = toc;

    % Evaluate on test data
    test_output = sigmoid(sigmoid(X_test * W1 + repmat(b1, size(X_test, 1), 1)) * W2 + repmat(b2, size(X_test, 1), 1));
    test_predictions = test_output >= 0.5;  % 0 or 1 predictions
    test_accuracy = mean(test_predictions == y_test) * 100;  % Accuracy as a percentage
end

% Experiment with different learning rates, hidden neurons, and epochs
learning_rates = [0.001, 0.01, 0.1];
hidden_neurons_list = [10, 20];
epochs = 1000;

% Initialize lists to store results for plotting
all_losses = [];
all_accuracies = [];
all_training_times = [];

% Loop through each combination of hyperparameters and train the model
for learning_rate = learning_rates
    for hidden_neurons = hidden_neurons_list
        fprintf('\nTraining with Learning Rate = %.4f, Hidden Neurons = %d\n', learning_rate, hidden_neurons);
        
        % Train the neural network
        [losses, test_accuracy, training_time] = train_neural_network(X_train, y_train, X_test, y_test, learning_rate, hidden_neurons, epochs);
        
        % Store results for plotting
        all_losses = [all_losses; losses'];
        all_accuracies = [all_accuracies; test_accuracy];
        all_training_times = [all_training_times; training_time];
        
        % Plot the training loss
        plot(1:epochs, losses, 'DisplayName', sprintf('LR=%.4f, Hidden=%d neurons', learning_rate, hidden_neurons));
        hold on;
        
        % Output the final test accuracy
        fprintf('Test Accuracy: %.2f%%\n', test_accuracy);
        fprintf('Training Time: %.2f seconds\n', training_time);
    end
end

% Display the loss plot for all experiments
xlabel('Epochs');
ylabel('Training Loss');
title('Training Loss vs Epochs for Different Learning Rates and Hidden Neurons');
legend show;
hold off;

% Plot the test accuracies
figure;
plot(all_accuracies, 'o-');
xlabel('Experiment Index');
ylabel('Test Accuracy (%)');
title('Test Accuracy for Different Learning Rates and Hidden Neurons');

% Plot the training times
figure;
plot(all_training_times, 'o-', 'Color', 'r');
xlabel('Experiment Index');
ylabel('Training Time (seconds)');
title('Training Time for Different Learning Rates and Hidden Neurons');
