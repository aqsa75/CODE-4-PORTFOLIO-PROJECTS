README for ML Project Code

Overview:

This project involves training and evaluating two machine learning models (Logistic Regression and Random Forest) using a dataset (first_ML.csv). The code includes hyperparameter tuning for both models, final model evaluation, and performance metrics generation.
The models are trained using 70% of the data, and the remaining 30% is used as the test set for evaluation.

Requirements:
MATLAB Version: R2023b or later (earlier versions will most likely work as well)

Toolboxes:
Statistics and Machine Learning Toolbox.

Libraries:
No external libraries are required; the necessary functions are built into MATLAB.

File Structure:
- first_ML.csv: The dataset used in this project. Change the path of this file.
- Initial.m: The initial script containing the model splitting and metrics before optimization.
- Final.m: The main script containing the model training and evaluation after cross validation and tuning.

Instructions for Running the Code:
The dataset used in this project is first_ML.csv. If you do not have this file locally, you can download it from the zip file 

Open MATLAB and run both of the matlab files: Initial.m and Final.m scripts.
 The scripts will:
- Load and preprocess the data.
- Split the data into training and test sets (70:30).
- Perform hyperparameter tuning for Logistic Regression and Random Forest models using cross-validation.
- Train the final models using the best hyperparameters.
- Evaluate the models on the test set and output performance metrics.
- Plot performance figures (ROC curves, confusion matrices, etc.).

Dependencies:
- No external dependencies are needed.
- MATLAB's built-in functions and the Statistics and Machine Learning Toolbox are used.

Test Set:
The dataset is split into 70% training and 30% test sets. The test set is used for final model evaluation.
The test set is stored in the variables X_test (features) and Y_test (target).

Key Outputs:
- The script will display: Performance Metrics, Confusion Matrices, ROC Curves and a Bar Chart Comparison

Notes:
- Ensure you have the required MATLAB toolboxe installed (Statistics and Machine Learning Toolbox).

