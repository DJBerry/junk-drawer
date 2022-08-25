'''
Functions and classes to help with recording information on model training,
testing, and calculating model performance.

    * generate_model_name: Function that takes in information about the
    model and turns it into a unique, descriptive name.

    * ModelPerformanceReporter: Class that generates a reporter object which
    takes in information on the training and testing process, model predic-
    tions, actual test values, and similar, and can then generate two different
    reports. One is a summary containing the model's mean absolute percent
    error, the spread of errors, and a measure of confidence interval
    performance, along with information on the train-test process. The other
    is a detailed report containing all untransformed test inputs, associated
    model predictions for target and confidence bounds, and actual target values,
    along with information on the train-test process. This is second report is
    useful for detailed analysis of where a model might be struggling.

    * RunSummary: Class that can be used to gather summaries when multiple
    train-test iterations are carried out. This can occur when evaluating a
    number of different model settings for comparison (necessitating multiple
    rounds of k-folds cross-validation), when there are multiple models trained
    and tested in a single loop, when a model is tested against multiple data-
    sets in a single loop, etc. Will output a single CSV file to a defined
    folder location when the run is completed.
'''

from datetime import datetime

import pandas as pd
import numpy as np


def generate_model_name(model_type: str, target: str, description: list, sep = "-"):
    '''
    Generate a descriptive name for the current model being trained and tested.
    The name will start with the current date and time down to a fraction of a
    second to help with sorting and uniqueness. If a greater guarantee of
    uniqueness is needed, a UUID may be a better choice.

    Parameters
    ----------
    model_type: str
        The type of model being built, such as GPR, linear regression, etc.

    target: str
        Name of the target.

    description: list
        A description of the model being trained and tested. This can include
        important settings or information about the train-test process that
        differentiate this model from others. Examples might include Gaussian
        noise setting, data scaling method, kernel, etc. Entries in the list
        will be concatenated together to form part of the model name.

    sep: "-"
        Text symbol to use as a separator when concatenating elements to form
        the model name. The separator is automatically doubled between major
        sections corresponding to the function parameters. For example:
        "2022-08-23--13-05-23.1234--gpr--price--kernel-RBF-White-scale-standard".

    Returns
    -------
    model_name: str
    '''

    # String to hold elements of the name
    name_lst = []

    # Get current date and time down to a fraction of a second
    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")
    name_lst.append(date_time + sep)

    # Add model type
    name_lst.append(model_type + sep)

    # Add target to name list
    name_lst.append(target + sep)

    # Ensure elements of description are strings and add to name
    for element in description:
        name_lst.append(str(element))

    # Build model name
    model_name = sep.join(name_lst)

    return model_name


class ModelPerformanceReporter:
    '''
    Record model predictions and calculate performance.

    Parameters
    ----------
    model_name: str
        The name of the current model being trained or tested.

    model_info: pd.DataFrame
        A dataframe made up of a single row plus column names where each column
        contains an important data point about the training and testing process
        for the current model. Examples might include number of training points,
        number of folds, noise setting, how the data is scaled, kernel choice,
        and any other settings which could impact how the model functions.
    '''

    def __init__(self, model_name: str, model_info: pd.DataFrame):
        # Track model predictions and inputs
        self.preds = []  # Target predictions
        self.stds = []  # Predicted standard deviations
        self.ci_ub = []  # Predicted upper confidence bounds
        self.ci_lb = []  # Predicted lower confidence bounds
        self.X_test_inputs = pd.DataFrame()  # Inputs used for prediction

        # Track testing values
        self.act_vals = []  # Actual values
        self.fold_nums = []  # Track fold number
        self.test_indices = []  # Track indices used to select test instances

        # Model results
        self.percent_error_description = pd.DataFrame()  # summary of model error
        self.ci_tracker = []  # Track if actual values fall in confidence bounds
        self.ci_performance = 0

        # Model info and summary
        self.model_name = model_name  # Name of model
        self.model_info = model_info  # Model settings
        self.results_summary = pd.DataFrame()  # Summary of results

        # Detailed model results
        self.predictions_and_actual_vals = pd.DataFrame()
        self.detailed_results = pd.DataFrame()

    def record_predictions(self,
                           predictions: list or np.array,
                           actual_values: list or np.array,
                           standard_devs: list or np.array,
                           upper_bounds: list or np.array,
                           lower_bounds: list or np.array):
        '''
        Record model predictions and actual target values.

        Parameters
        ----------
        predictions: list or np.array
            A list or array containing model predictions of the target value.

        actual_values: list or np.array
            A list or array containing actual target values.

        standard_devs: list or np.array
            A list or array containing predicted standard deviations.

        upper_bounds: list or np.array
            A list or array containing predicted upper bounds.

        lower_bounds:
            A list or array containing predicted lower bounds.
        '''
        self.preds.append(predictions)
        self.act_vals.append(actual_values)
        self.stds.append(standard_devs)
        self.ci_ub.append(upper_bounds)
        self.ci_lb.append(lower_bounds)

    def record_testing_info(self,
                            fold_number: int,
                            test_indices: list or np.array,
                            X_inputs: pd.DataFrame):
        '''
        Record information on testing the model.

        Parameters
        ----------
        fold_number: int
            The current fold number.

        test_indices: list or np.array
            A list or array containing the indices used to select the test
            data. These can be used to compare the recorded test inputs with
            what is in the original dataset.

        X_inputs: pd.DataFrame
            The untransformed inputs used for this set of tests.
        '''
        # Record current fold
        self.fold_nums.extend([fold_number]*len(test_indices))

        # Record indices used to select the test data
        self.test_indices.extend(test_indices)

        # Record test inputs before transforming them
        self.X_test_inputs = pd.concat([self.X_test_inputs, pd.DataFrame(X_inputs)],
                                       axis=0, ignore_index=True)

    def get_performance_summary(self):
        '''
        Get a summary of model performance including mean absolute percent error
        and confidence interval performance.

        Returns
        -------
        self.results_summary: pd.DataFrame
            A dataframe containing mean absolute percent error, the percentage
            of actual values captured by the predicted confidence bounds, and
            other model performance metrics.
        '''
        # Flatten arrays to allow for calculation
        self.preds = self._clean_subarrays(self.preds)
        self.stds = self._clean_subarrays(self.stds)
        self.ci_ub = self._clean_subarrays(self.ci_ub)
        self.ci_lb = self._clean_subarrays(self.ci_lb)
        self.act_vals = self._clean_subarrays(self.act_vals)

        # Calculate mean absolute percent error metrics
        percent_error = pd.DataFrame(100 * (np.abs(self.preds - self.act_vals)) / self.act_vals)
        self.percent_error_description = percent_error.describe()

        # Loop through actual values and determine if they fall between
        # predicted upper and lower confidence bounds
        for y, lb, ub in zip(self.act_vals, self.ci_lb, self.ci_ub):
            if y >= lb and y <= ub:
                self.ci_tracker.append(1)
            else:
                self.ci_tracker.append(0)

        # Calculate percentage of actual values that fall between predicted
        # upper and lower confidence bounds
        self.ci_performance = 100 * sum(self.ci_tracker)/len(self.ci_tracker)

        # Add confidence interval performance and model name
        model_perf = self.percent_error_description.copy().transpose()
        model_perf['Confidence_Interval_Performance'] = self.ci_performance
        model_perf['Model_Name'] = self.model_name

        # Gather model summary and output
        self.results_summary = model_perf.join(
            self.model_info, lsuffix='_caller', rsuffix='_other')

        return self.results_summary

    def get_detailed_results(self):
        '''
        Get model predictions, actual test values, and other detailed
        training and testing data for each instance in the test set.

        Returns
        -------
        self.detailed_results: pd.DataFrame
            A pd.DataFrame containing all predicted values, target values, and
            inputs, as well as information for tracking like model name.
        '''
        # Gather actual values and predictions
        self.predictions_and_actual_vals = pd.DataFrame({
            'Actual_Values': list(self.act_vals),
            'Predicted_Values': list(self.preds),
            'Predicted_Upper_Confidence_Bound': list(self.ci_ub),
            'Predicted_Lower_Confidence_Bound': list(self.ci_lb),
            'Confidence_Interval_Tracker': list(self.ci_tracker),
            'Model_Name': [self.model_name]*len(self.act_vals)
        })

        # Join model inputs with targets and predictions
        self.detailed_results = self.X_test_inputs.join(
            self.predictions_and_actual_vals, lsuffix='_caller', rsuffix='_other')

        return self.detailed_results

    def _check_for_subarrays(self, array: list or np.array):
        '''
        Determine if an array or list contains sub-arrays or sub-lists.

        Parameters
        ----------
        array: list or np.array
            A list or array that you want to check for sub-lists or sub-arrays.

        Returns
        -------
        needs_cleaning: boolean
            Returns 'True' if an element in the list or array causes
            isinstance() to return true when compared with np.ndarray or list.
        '''
        for item in array:
            if isinstance(item, np.ndarray) or isinstance(item, list):
                needs_cleaning = True
                break
            else:
                needs_cleaning = False

        return needs_cleaning

    def _clean_subarrays(self, array: list or np.array):
        '''
        Flattens lists or arrays which contain sub-lists or sub-arrays.

        Parameters
        ----------
        array: list or np.array
            A list or array that you want to check to see if it contains
            sub-lists or sub-arrays.

        Returns
        -------
        array: np.array
            A flattened array of dtype='float64'.
        '''
        # Check if array needs to be cleaned
        needs_cleaning = self._check_for_subarrays(array)

        # Flatten array of arrays
        while needs_cleaning:
            temp = []
            for i in range(len(array)):
                cool = np.array(array[i]).ravel()
                for j in cool:
                    temp.append(j)

            # Check if array still needs cleaning
            array = np.array(temp, dtype='float64')
            needs_cleaning = self._check_for_subarrays(array)

        return np.array(array, dtype='float64')


class RunSummary:
    '''
    This class creates a summarizer that gathers performance results over many
    iterations and output a summary at the end. This is useful when testing
    multiple models or settings during a single run.

    Parameters
    ----------
    output_location: str
        Folder location where to output csv containing summary.
    '''

    def __init__(self, output_location: str) -> None:
        # Output location
        self.output_location = output_location
        self.summaries_location = output_location

        # Model summaries
        self.run_summaries = pd.DataFrame()

        # Run ID
        self.run_id = self._generate_run_id()

    def record_model_performance(self, new_model_summary: pd.DataFrame):
        '''
        Adds a new summary to the current summarizer. Functions by concatenating
        a new dataframe to one which contains earlier summaries.

        Parameters
        ----------
        new_model_summary: pd.DataFrame
            A dataframe summarizing the performance of the model. This summary
            will be concatenated to other summaries in the summarizer, so it
            should have identical columns.
        '''
        self.run_summaries = pd.concat([self.run_summaries, new_model_summary],
                                       ignore_index=True)

    def output_run_summaries_csv(self):
        '''
        Outputs a csv file containing the model performance summaries to the
        previously defined folder location. File name is the id generated by
        the class.
        '''
        fname = self.output_location + "\\" + self.run_id + ".csv"
        self.run_summaries.to_csv(fname)

    def _generate_run_id(self):
        '''
        Generate an id for the current train-test run. Uses the current date
        and time down to a fraction of a second to help with uniqueness.
        '''

        # Get date and time for current run
        date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")

        # Generate Run ID
        name = 'Train-Test-Run--' + date_time

        return name
