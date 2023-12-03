import pandas as pd
import scipy.stats as stats
import numpy as np


class StatisticalTestRunner:
    """
    This class provides various statistical tests and analysis methods.
    It is designed to make running common statistical tests more convenient.
    """

    @staticmethod
    def get_feature_type(data, column):
        """
        Determines the type of a feature (column) in a given dataset.

        Args:
        data (pd.DataFrame): The dataset containing the feature.
        column (str): The name of the column to determine the type of.

        Returns:
        str: Type of the feature - "numeric" or "categorical".
        """

        if pd.api.types.is_numeric_dtype(data[column]):
            return 'numerical'
        else:
            return 'categorical'
        
    
    # __________________________________________________________________________________________Manoj
    # __________________________________________________________________________________________Ganesh
    @staticmethod
    def analyze_features(data, target_variable, alpha=0.05):
        """
        Analyze features in a DataFrame against a target variable. Drops features 
        which are not significant and prints out significant and non-significant features.

        :param data: Pandas DataFrame containing the features and target variable.
        :param target_variable: The name of the target variable.
        :param alpha: Significance level, defaults to 0.05.
        :return: DataFrame with only significant features.
        """
        significant_features = []
        non_significant_features = []

        for feature in data.columns:
            if feature != target_variable:
                test_name = StatisticalTestRunner.determine_statistical_test(
                    data, feature, target_variable)
                result = StatisticalTestRunner.execute_statistical_test(
                    data, feature, target_variable)

                if result.get('p_value', 1) < alpha:
                    significant_features.append(feature)
                else:
                    non_significant_features.append(feature)

        # Printing the results
        print("Significant features:", significant_features)
        print("Non-significant features:", non_significant_features)

        return data[significant_features + [target_variable]]
