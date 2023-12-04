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
    
    @staticmethod
    def determine_statistical_test(data, feature1, feature2=None, paired=False):
        """
        Determine the appropriate statistical test based on the data and the names of one or two features.

        :param data: The dataset (pandas DataFrame).
        :param feature1: The name of the first feature (column).
        :param feature2: The name of the second feature (column), optional.
        :param paired: Boolean indicating if the samples are paired, applicable for two numerical features.
        :return: Name of the statistical test.
        """
        feature1_type = StatisticalTestRunner.get_feature_type(data, feature1)
        feature2_type = StatisticalTestRunner.get_feature_type(
            data, feature2) if feature2 else None
        sample_size = len(data)

        if feature2_type is None:
            # Single feature analysis
            if feature1_type == 'categorical':
                return "Chi-squared goodness of fit test"
            elif feature1_type == 'numerical':
                return "One-sample t-test" if sample_size <= 30 else "Z-test"

        elif feature1_type == feature2_type:
            # Both features are of the same type
            if feature1_type == 'categorical':
                return "Chi-squared test of independence"
            elif feature1_type == 'numerical':
                if paired:
                    return "Paired t-test"
                elif sample_size > 30:
                    return "Two-sample t-test"
                else:
                    return "ANOVA"

        else:
            # Features are of different types
            if feature1_type == 'numerical' and feature2_type == 'categorical':
                return "ANOVA"
            elif feature1_type == 'categorical' and feature2_type == 'numerical':
                return "ANOVA"

        return "Unable to determine an appropriate test"
        
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
