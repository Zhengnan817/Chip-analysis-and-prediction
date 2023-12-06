import pandas as pd


class DataProcess:
    """
    A class for handling and processing data from a CSV file using Pandas.

    This class provides methods to initialize the class with a CSV file, view the data,
    check for null values, and remove rows with null values.

    Attributes:
    - filepath (str): The path to the CSV file.
    - df (pd.DataFrame): The Pandas DataFrame containing the data from the CSV file.

    Methods:
    - __init__(self, filepath): Initialize the Data_process class with a filepath and read the CSV file into a Pandas DataFrame.
    - view_data(self): Display the first few rows of the dataframe.
    - check_data(self): Print information about the file and display the first row of the dataframe.
    - remove_null(self): Remove rows with null values from the dataframe.
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
        self.df['Release Date'] = pd.to_datetime(
            self.df['Release Date'], format='%m/%d/%y'
        )
        return self.df

    def handle_missing_value(self):
        self.df = self.df.dropna(subset=['Process Size', 'Release Date'])
