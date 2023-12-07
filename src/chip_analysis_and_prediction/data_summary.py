"""
Module Name: data_summary
Description: This module contains functions and classes related to process data.
Author: Zhengnan Li
Date: December 6, 2023
"""
import pandas as pd


class DataProcess:
    """
    A class for processing and cleaning data from a CSV file using Pandas.

    Attributes:
    - filepath (str): The path to the CSV file.
    - df (pandas.DataFrame): The Pandas DataFrame containing the data from the CSV file.
    """

    def __init__(self, filepath):
        """
        Initialize the Solution class with a filepath and read the CSV file into a Pandas DataFrame.

        Parameters:
        - filepath (str): The path to the CSV file.
        """
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def view_data(self):
        """
        Display the first few rows of the dataframe.

        Returns:
        pandas.DataFrame: The first few rows of the dataframe.
        """
        return self.df

    def check_data(self):
        """
        Print information about the file and display the first row of the dataframe.
        """
        # return the number of null values for each column
        return self.df.isna().sum()

    def remove_null(self):
        """
        Remove rows with null values from the dataframe.

        Returns:
        pandas.DataFrame: The dataframe with null values removed.
        """
        # Drop rows with null values and return the updated dataframe
        return self.df.dropna()

    def data_clean(self):
        """
        Clean the data by converting 'Release Date' to datetime format.

        Returns:
        - pd.DataFrame: The cleaned DataFrame.
        """
        self.df['Release Date'] = pd.to_datetime(
            self.df['Release Date'], format='%m/%d/%y'
        )
        return self.df

    def handle_missing_value(self):
        """
        Handle missing values by dropping rows with NaN in 'Process Size' and 'Release Date'.

        Returns:
        - pd.DataFrame: The DataFrame after handling missing values.
        """
        self.df = self.df.dropna(subset=['Process Size', 'Release Date'])
