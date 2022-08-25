'''
Example for using report_model_performance.py classes and methods.

The report_model_performance.py module was originally developed as part of a
research project that involved comparing the effectiveness of different machine
learning approaches when trained and tested on the same data. Because this
involved running multiple train-test iterations for different types of models
and model settings, setting up and tweaking all the lists, dataframes, and
performance metrics to capture the detailed information necessary became a
major time sink. The methods and classes in report_model_performance.py
attempted to address these concerns by:

    * Including a method to generate unique but still meaningful names
    that could be used to differentiate models.

    * Creating a class that captures the training and testing information
    for a single model. This class, ModelPerformanceReporter, records testing
    inputs, predictions, predicted confidence bounds, and similar during the
    train-test process, and at the end returns a summary with performance
    metrics as well as a detailed report documenting the inputs for each data
    instance, the associated model predictions, and various settings the
    researcher wants to track.

    * Allowing for the creation of a single, large summary when iterating
    through multiple rounds of training and testing, such as when experimenting
    with combinations of various noise levels, model kernels, etc. The
    RunSummary class captures performance summaries from multiple instance of
    ModelPerformanceReporter and combines them into a single summary for the
    entire run.

While the below example is somewhat contrived, and the models themselves don't
perform particularly well, it still demonstrates the module's ability to
capture relevant training and testing data as well as output detailed
information on model performance.
'''

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn import datasets

from report_model_performance import generate_model_name
from report_model_performance import ModelPerformanceReporter
from report_model_performance import RunSummary


def cross_validation(X, y, gpr_settings, mlp_settings):
    # Generate name for gpr model
    gpr_dscrpt = [gpr_settings['kernel'], gpr_settings['alpha']]
    gpr_name = generate_model_name('gpr', 'diabetes_y', gpr_dscrpt)

    # Generate name for mlp model
    mlp_dscrpt = [mlp_settings['alpha'],
                  mlp_settings['activation'],
                  mlp_settings['learning_rate']]
    mlp_name = generate_model_name('mlp', 'diabetes_y', mlp_dscrpt)

    # Create a reporter for gpr performance
    gpr_reporter = ModelPerformanceReporter(gpr_name,
                                            pd.DataFrame(gpr_settings, index=[0]))

    # Create a reporter for mlp performance
    mlp_reporter = ModelPerformanceReporter(mlp_name,
                                            pd.DataFrame(mlp_settings, index=[0]))

    # Define number of folds
    kf = KFold(n_splits=10)

    # Variable for tracking the current fold
    fold_num = 0

    # Perform k-fold cross-validation
    for train, test in kf.split(y):
        # Track current fold number
        fold_num = fold_num + 1

        # Get train and test data
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        # Record train and test information
        gpr_reporter.record_testing_info(fold_number=fold_num,
                                         test_indices=test,
                                         X_inputs=X_test)

        mlp_reporter.record_testing_info(fold_number=fold_num,
                                         test_indices=test,
                                         X_inputs=X_test)

        # Scale the training data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Apply the same scaler to the test data to avoid data leakage
        X_test = scaler.transform(X_test)

        # Create and train Gaussian process regression model
        gpr = GaussianProcessRegressor(kernel=gpr_settings['kernel'],
                                       alpha=gpr_settings['alpha'],
                                       optimizer=gpr_settings['optimizer'],
                                       n_restarts_optimizer=gpr_settings['n_restarts_optimizer'],
                                       normalize_y=gpr_settings['normalize_y'],
                                       copy_X_train=gpr_settings['copy_X_train'])
        gpr.fit(X_train, y_train)

        # Create and train Multi-layer Perceptron regressor model
        mlp = MLPRegressor(activation=mlp_settings['activation'],
                           alpha=mlp_settings['alpha'],
                           learning_rate=mlp_settings['learning_rate'])
        mlp.fit(X_train, y_train)

        # Test Gaussian process regression model
        gpr_preds, gpr_std = gpr.predict(X_test, return_std=True)

        # Get predicted upper and lower bounds from Gaussian process regression
        gpr_upper_bounds = gpr_preds + 1.96*gpr_std
        gpr_lower_bounds = gpr_preds - 1.96*gpr_std

        # Record Gaussian process regression predictions
        gpr_reporter.record_predictions(predictions=gpr_preds,
                                        actual_values=y_test,
                                        standard_devs=gpr_std,
                                        upper_bounds=gpr_upper_bounds,
                                        lower_bounds=gpr_lower_bounds)

        # Test Multi-layer Perceptron regressor model
        mlp_preds = mlp.predict(X_test)

        # Record Multi-layer Perceptron regressor predictions
        mlp_reporter.record_predictions(predictions=mlp_preds,
                                        actual_values=y_test,
                                        standard_devs= [0]*len(y_test),
                                        upper_bounds=[0]*len(y_test),
                                        lower_bounds=[0]*len(y_test))

    return gpr_reporter, mlp_reporter

if __name__ == '__main__':

    # Location to put detailed model results
    detailed_results_loc = "C:\\Repositories\\junk-drawer\\Results_folder_test\\"

    # Location to put summaries of model performance
    run_summaries_loc = "C:\\Repositories\\junk-drawer\\Results_folder_test\\"

    # Import Scikit-Learn's Diabetes dataset
    X, y = datasets.load_diabetes(return_X_y=True)

    # Specify settings to iterate through
    all_settings = [
        [RBF(), 1e-5, 'logistic', 'constant'],
        [RBF(), 1e-10, 'logistic', 'adaptive'],
        [RBF()+WhiteKernel(), 1e-10, 'tanh', 'constant'],
        [RBF()+WhiteKernel(), 1e-10, 'tanh', 'adaptive']
    ]

    # Create a summarizer to gather results on each model tested
    run_summarizer = RunSummary(output_location=run_summaries_loc)

    # Iterate through settings
    for settings in all_settings:

        gpr_settings = {
            'kernel': settings[0],
            'alpha': settings[1],
            'optimizer': 'fmin_l_bfgs_b',
            'n_restarts_optimizer': 0,
            'normalize_y': False,
            'copy_X_train': True,
            'random_state': 'None',
            'activation': 'N/A',
            'learning_rate': 'N/A',
            'data_scaling': 'standard scaler',
            'other_settings': 'N/A'
        }

        mlp_settings = {
            'kernel': 'N/A',
            'alpha': settings[1],
            'optimizer': 'N/A',
            'n_restarts_optimizer': 'N/A',
            'normalize_y': 'N/A',
            'copy_X_train': 'N/A',
            'random_state': 'N/A',
            'activation': settings[2],
            'learning_rate': settings[3],
            'data_scaling': 'standard scaler',
            'other_settings': 'Defaults'
        }

        # Perform k-fold cross validation
        gpr_reporter, mlp_reporter = cross_validation(X,
                                                      y,
                                                      gpr_settings,
                                                      mlp_settings)

        # Get Gaussian process regression model performance
        gpr_performance = gpr_reporter.get_performance_summary()

        # Get detailed report on Gaussian process regression performance
        gpr_details = gpr_reporter.get_detailed_results()
        gpr_details.to_csv(
            detailed_results_loc + gpr_reporter.model_name + '.csv')

        # Get Multi-layer Perceptron regressor model performance
        mlp_performance = mlp_reporter.get_performance_summary()
        mlp_performance.to_csv(
            detailed_results_loc + mlp_reporter.model_name + '.csv')

        # Record model performance summaries
        run_summarizer.record_model_performance(gpr_performance)
        run_summarizer.record_model_performance(mlp_performance)

    # Output summaries from model runs
    run_summarizer.output_run_summaries_csv()