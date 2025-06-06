% % Load the dataset
df = readtable('/Users/aqsa/Desktop/NC-G/churndata.csv');

% Assuming df is your dataset
% Check the type of the Geography columns to understand their current type
disp(class(df.Geography_France));  % Should display 'cell' or 'double'

% Convert 'Geography_France', 'Geography_Germany', and 'Geography_Spain' to logical (1/0) if they are of type 'cell'
if iscell(df.Geography_France)
    df.Geography_France = double(strcmp(df.Geography_France, 'France'));  % Convert 'France' to 1, others to 0
end
if iscell(df.Geography_Germany)
    df.Geography_Germany = double(strcmp(df.Geography_Germany, 'Germany'));  % Convert 'Germany' to 1, others to 0
end
if iscell(df.Geography_Spain)
    df.Geography_Spain = double(strcmp(df.Geography_Spain, 'Spain'));  % Convert 'Spain' to 1, others to 0
end

% Convert all other necessary columns to double if they are not already
df.Gender = double(df.Gender);  % Assuming this is already numeric
df.HasCrCard = double(df.HasCrCard);  % Assuming this is already numeric
df.IsActiveMember = double(df.IsActiveMember);  % Assuming this is already numeric

% Select features and target
X = df(:, {'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', ...
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', ...
    'Geography_France', 'Geography_Germany', 'Geography_Spain'});  % Features
y = df.Exited;  % Target (binary classification)

% Split into train-test sets (80% train, 20% test)
cv = cvpartition(height(df), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv));
y_test = y(test(cv));

% Convert tables to arrays (matrix)
X_train = table2array(X_train);
X_test = table2array(X_test);

% Sigmoid activation and derivative functions
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end

function out = sigmoid_derivative(x)
    out = x .* (1 - x);
end

% Function to create and train the model
function [losses, accuracies, training_times] = train_neural_network(X_train, y_train, X_test, y_test, learning_rate, hidden_neurons, epochs)
    input_size = size(X_train, 2);  % Number of input features
    output_size = 1;  % Output neuron (binary classification)

    % Initialize weights and biases
    W1 = randn(input_size, hidden_neurons) * 0.01;
    b1 = zeros(1, hidden_neurons);
    W2 = randn(hidden_neurons, output_size) * 0.01;
    b2 = zeros(1, output_size);

    % Initialize loss, accuracy, and training time tracking
    losses = [];
    accuracies = [];
    training_times = [];

    % Training loop
    for epoch = 1:epochs
        epoch_start = tic; % Start timer for current epoch

        % Forward pass
        Z1 = X_train * W1 + repmat(b1, size(X_train, 1), 1);  % (m x n) * (n x h) + (m x h)
        A1 = sigmoid(Z1);  % (m x h)
        Z2 = A1 * W2 + repmat(b2, size(X_train, 1), 1);  % (m x h) * (h x 1) + (m x 1)
        A2 = sigmoid(Z2);  % Output layer prediction (m x 1)

        % Calculate loss (Binary Cross-Entropy)
        loss = -mean(y_train .* log(A2) + (1 - y_train) .* log(1 - A2));
        losses = [losses; loss];

        % Accuracy Calculation (for test set)
        test_output = sigmoid(sigmoid(X_test * W1 + repmat(b1, size(X_test, 1), 1)) * W2 + repmat(b2, size(X_test, 1), 1));
        test_predictions = test_output >= 0.5;  % 0 or 1 predictions
        accuracy = mean(test_predictions == y_test) * 100;  % Accuracy as a percentage
        accuracies = [accuracies; accuracy];

        % Backpropagation
        dA2 = A2 - y_train;  % (m x 1)
        dZ2 = dA2 .* sigmoid_derivative(A2);  % (m x 1)
        dW2 = A1' * dZ2;  % (h x m)' * (m x 1) -> (h x 1)
        db2 = sum(dZ2, 1);  % (1 x 1)

        dA1 = dZ2 * W2';  % (m x 1) * (1 x h) -> (m x h)
        dZ1 = dA1 .* sigmoid_derivative(A1);  % (m x h)
        dW1 = X_train' * dZ1;  % (n x m)' * (m x h) -> (n x h)
        db1 = sum(dZ1, 1);  % (1 x h)

        % Update weights and biases using gradient descent
        W1 = W1 - learning_rate * dW1;  % (n x h) - (n x h)
        b1 = b1 - learning_rate * db1;  % (1 x h) - (1 x h)
        W2 = W2 - learning_rate * dW2;  % (h x 1) - (h x 1)
        b2 = b2 - learning_rate * db2;  % (1 x 1) - (1 x 1)

        % Calculate the time taken for the current epoch
        epoch_time = toc(epoch_start);
        training_times = [training_times; epoch_time];

        % Print progress every 100 epochs (can adjust this if needed)
        if mod(epoch, 100) == 0
            fprintf('Epoch %d: Loss = %.4f\n', epoch, loss);
        end
    end

    % Print final results after training completes
    test_output_final = sigmoid(sigmoid(X_test * W1 + repmat(b1, size(X_test, 1), 1)) * W2 + repmat(b2, size(X_test, 1), 1));
    final_predictions = test_output_final >= 0.5;  % 0 or 1 predictions
    final_accuracy = mean(final_predictions == y_test) * 100;  % Final Test Accuracy

    fprintf('\nTest Accuracy: %.2f%%\n', final_accuracy);
    fprintf('Training Time: %.2f seconds\n', sum(training_times));  % Total training time
end

% Experiment with different hidden neurons and epochs
learning_rate_list = [0.001, 0.01, 0.1];  % Learning rates
hidden_neurons_list = [10, 20, 50];  % Number of hidden neurons
epochs = 1000;  % Fixed number of epochs

% Loop through each learning rate and hidden neuron configuration and train the model
for learning_rate = learning_rate_list
    for hidden_neurons = hidden_neurons_list
        fprintf('\nTraining with Learning Rate = %.4f, Hidden Neurons = %d\n', learning_rate, hidden_neurons);
        
        % Train the neural network
        [losses, accuracies, training_times] = train_neural_network(X_train, y_train, X_test, y_test, learning_rate, hidden_neurons, epochs);
    end
end
