import pandas as pd
import scipy.stats as stats
import numpy as np


@staticmethod
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

    @staticmethod
    def execute_statistical_test(data, feature1, feature2=None, detailed=False, handle_missing='remove'):
        """
        Execute the appropriate statistical test based on the test name and input features.

        :param data: The dataset (pandas DataFrame).
        :param feature1: The name of the first feature (column).
        :param feature2: The name of the second feature (column), optional.
        :param test_name: The name of the test to be executed.
        :param detailed: Boolean to determine if detailed results are needed.
        :param handle_missing: Strategy for handling missing data ('remove' or 'impute').
        :return: Test result (summary or detailed).
        """
    # Handle missing data
        if handle_missing == 'remove':
            data = data.dropna(subset=[feature1] if feature2 is None else [
                               feature1, feature2])
        elif handle_missing == 'impute':
            for feature in [feature1, feature2]:
                if feature and pd.api.types.is_numeric_dtype(data[feature]):
                    data[feature].fillna(data[feature].mean(), inplace=True)
                elif feature:
                    data[feature].fillna(data[feature].mode()[0], inplace=True)

        # Helper functions for each test
        def perform_z_test(sample):
            mean = sample.mean()
            std = sample.std(ddof=1) / len(sample)**0.5
            z_score = mean / std
            p_value = stats.norm.sf(abs(z_score)) * 2
            return {"z_score": z_score, "p_value": p_value}

        def perform_one_sample_t_test(sample):
            population_mean = data[feature1].mean()
            t_stat, p_value = stats.ttest_1samp(sample, population_mean)
            return {"t_statistic": t_stat, "p_value": p_value}

        def perform_two_sample_t_test(sample1, sample2):
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            return {"t_statistic": t_stat, "p_value": p_value}

        def perform_paired_t_test(sample1, sample2):
            t_stat, p_value = stats.ttest_rel(sample1, sample2)
            return {"t_statistic": t_stat, "p_value": p_value}

        def perform_anova(data, numeric_feature, categorical_feature):
            groups = data.groupby(categorical_feature)[
                numeric_feature].apply(list)
            f_stat, p_value = stats.f_oneway(*groups)
            return {"f_statistic": f_stat, "p_value": p_value}

        def perform_chi_squared_test(table):
            stat, p, dof, expected = stats.chi2_contingency(table)
            return {"chi2_statistic": stat, "p_value": p, "degrees_of_freedom": dof, "expected_frequencies": expected}

        # Determine the test to be used
        test_name = StatisticalTestRunner.determine_statistical_test(
            data, feature1, feature2)

        result = {}
        if test_name == "Z-test":
            result = perform_z_test(data[feature1])
        elif test_name == "One-sample t-test":
            result = perform_one_sample_t_test(data[feature1])
        elif test_name == "Two-sample t-test":
            result = perform_two_sample_t_test(data[feature1], data[feature2])
        elif test_name == "Paired t-test":
            result = perform_paired_t_test(data[feature1], data[feature2])
        elif test_name == "ANOVA":
            if feature2 is not None:
                # Identify which feature is numeric and which is categorical
                if pd.api.types.is_numeric_dtype(data[feature1]) and not pd.api.types.is_numeric_dtype(data[feature2]):
                    numeric_feature = feature1
                    categorical_feature = feature2
                elif pd.api.types.is_numeric_dtype(data[feature2]) and not pd.api.types.is_numeric_dtype(data[feature1]):
                    numeric_feature = feature2
                    categorical_feature = feature1
                else:
                    return "Error: ANOVA requires one numeric and one categorical feature."

                # Perform ANOVA
                result = perform_anova(
                    data, numeric_feature, categorical_feature)
            else:
                return "Error: ANOVA requires two features."
        elif test_name == "Chi-squared test of independence":
            contingency_table = pd.crosstab(data[feature1], data[feature2])
            result = perform_chi_squared_test(contingency_table)

        # Return detailed or summary result based on user preference
        if detailed:
            summary = {
                "test_name": test_name,
                "test_outcome": "Significant" if result.get("p_value", 1) < 0.05 else "Not Significant",
                "p_value": result.get("p_value")
            }
            return summary

        else:
            return result
