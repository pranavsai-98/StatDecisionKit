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
