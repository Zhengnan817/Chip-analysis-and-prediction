"""
Module Name: exploratory_data_analysis
Description: This module contains functions and classes related to exploratory data analysis
Author: Zhengnan Li
Date: December 6, 2023
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class EDA:
    """
    A class for performing Exploratory Data Analysis (EDA) on Chips of CPU and GPU.

    This class provides methods to generate various visualizations for analyzing the distribution
    and characteristics of data, focusing on 'Type', 'Vendor', 'Process Size', 'Vendor Distribution', and 'TDP'.

    Attributes:
    - df (pd.DataFrame): The input DataFrame for analysis.

    Methods:
    - __init__(self, df): Initialize the EDA class with a DataFrame.
    - type_vendor_sns(self): Create a Seaborn barplot for 'Type' and 'Vendor'.
    - type_vendor_plt(self): Create a Matplotlib barplot for 'Type' and 'Vendor'.
    - process_size_sns(self): Create a Seaborn boxplot of chip process size.
    - process_size_plt(self): Create a Matplotlib boxplot of chip process size.
    - vendor_distribution_sns(self): Create a Seaborn countplot for the distribution of vendors.
    - vendor_distribution_plt(self): Create a Matplotlib barplot for the distribution of vendors.
    - TDP_distribution_sns(self): Create a Seaborn boxplot for the distribution of TDP in different vendors.
    - TDP_distribution_plt(self): Create a Matplotlib boxplot for the distribution of TDP in different vendors.
    """

    def __init__(self, df):
        """
        Initialize the EDA (Exploratory Data Analysis) class with a DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame for analysis.
        """
        self.df = df

    def type_vendor_sns(self):
        """
        Create a barplot for 'Type' and 'Vendor' in one picture using Seaborn.
        """
        # Create a barplot for 'Type' and 'Vendor' in one picture
        plt.figure(figsize=(12, 6))
        sns.catplot(
            data=self.df,
            x='Type',
            hue='Vendor',
            kind='count',
            palette='Set3',
            height=6,
            aspect=2,
        )

        # Set labels and title
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.title('Barplot of Type and Vendor')

        # Show the plot
        plt.show()

    def type_vendor_plt(self):
        """
        Create a barplot for 'Type' and 'Vendor' in one picture using Matplotlib.

        Returns:
        - None
        """

        # Create a barplot for 'Type' and 'Vendor' in one picture
        plt.figure(figsize=(12, 6))

        # Assuming self.df is your DataFrame
        grouped_data = self.df.groupby(['Type', 'Vendor']).size().unstack()

        # Plotting
        grouped_data.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 6))

        # Set labels and title
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.title('Barplot of Type and Vendor')

        # Show the plot
        plt.show()

    def process_size_sns(self):
        """
        Create a boxplot of chip process size using Seaborn.

        Returns:
        - None
        """
        # Create a Seaborn boxplot for chip process size
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=self.df['Process Size'], orient='h')
        # Set labels and title
        plt.title('Boxplot of chip process size')
        plt.xlabel('Values')
        # Show the plot
        plt.show()

    def process_size_plt(self):
        """
        Create a boxplot of chip process size using Matplotlib.

        Returns:
        - None
        """
        # Create a Matplotlib boxplot for chip process size
        plt.figure(figsize=(15, 10))
        plt.boxplot(self.df['Process Size'], vert=False)
        # Set labels and title
        plt.title('Boxplot of chip process size')
        plt.xlabel('Values')
        # Show the plot
        plt.show()

    def vendor_distribution_sns(self):
        """
        Create a countplot for the distribution of vendors using Seaborn.

        Returns:
        - None
        """
        # Create a Seaborn countplot for the distribution of vendors
        plt.figure(figsize=(12, 6))
        sns.countplot(
            x='Vendor', data=self.df, palette='husl', hue='Vendor', legend=False
        )
        plt.title('Distribution of Vendors')
        # Set labels and title
        plt.xlabel('Vendor')
        plt.ylabel('Count')
        plt.show()

    def vendor_distribution_plt(self):
        """
        Create a barplot for the distribution of vendors using Matplotlib.

        Returns:
        - None
        """
        # Create a Matplotlib barplot for the distribution of vendors
        plt.figure(figsize=(12, 6))

        # Assuming self.df["Vendor"] contains the vendor data
        unique_vendors, vendor_counts = np.unique(self.df['Vendor'], return_counts=True)

        plt.bar(unique_vendors, vendor_counts, color='mediumseagreen')
        # Set labels and title
        plt.title('Distribution of Vendors')
        plt.xlabel('Vendor')
        plt.ylabel('Count')
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        # Show the plot
        plt.show()

    def TDP_distribution_sns(self):
        """
        Create a boxplot for the distribution of TDP in different vendors using Seaborn.

        Returns:
        - None
        """
        # Create a Seaborn boxplot for the distribution of TDP in different vendors
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            x='Vendor',
            y='TDP',
            data=self.df,
            palette='Set3',
            hue='Vendor',
            legend=False,
        )
        # Set title
        plt.title('TDP distribute in different vendors')
        # Show the plot
        plt.show()

    def TDP_distribution_plt(self):
        """
        Create a boxplot for the distribution of TDP in different vendors using Matplotlib.

        Returns:
        - None
        """
        # Create a Matplotlib boxplot for the distribution of TDP in different vendors
        plt.figure(figsize=(8, 6))

        # Boxplot for TDP distribution in different vendors
        plt.boxplot(
            [
                self.df[self.df['Vendor'] == vendor]['TDP']
                for vendor in self.df['Vendor'].unique()
            ],
            labels=self.df['Vendor'].unique(),
            showfliers=False,
            patch_artist=True,
        )
        # Set labels and title
        plt.xlabel('Vendor')
        plt.ylabel('TDP')
        plt.title('TDP distribution in different vendors')
        # Show the plot
        plt.show()

    def process_size_vendor_sns(self):
        """
        Create a boxplot using seaborn to visualize the distribution of process sizes for different vendors.

        Parameters:
        - self: The instance of the class containing the DataFrame.

        Returns:
        None
        """
        # Create a boxplot using seaborn
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Vendor', y='Process Size', data=self.df)
        plt.title('Boxplot of Process Size for Different Vendors')
        plt.show()

    def process_size_vendor_plt(self):
        """
        Create a boxplot using matplotlib to visualize the distribution of process sizes for different vendors.

        Parameters:
        - self: The instance of the class containing the DataFrame.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        boxplot_data = [
            self.df['Process Size'][self.df['Vendor'] == vendor]
            for vendor in self.df['Vendor'].unique()
        ]
        plt.boxplot(boxplot_data, labels=self.df['Vendor'].unique())
        plt.title('Boxplot of Process Size for Different Vendors')
        plt.xlabel('Vendor')
        plt.ylabel('Process Size')
        plt.show()

    def process_size_date(self):
        """
        Create a scatter plot using seaborn to visualize the relationship between process size and release date.

        Parameters:
        - self: The instance of the class containing the DataFrame.

        Returns:
        None
        """
        # Assuming self.df is your DataFrame containing 'Process Size', 'Release Date', and 'Type' columns
        sns.scatterplot(data=self.df, x='Release Date', y='Process Size', hue='Type')
        plt.show()
