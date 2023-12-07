"""
This script performs Exploratory Data Analysis (EDA) on CPU and GPU chip data.

It uses the EDA class defined in the module to generate various visualizations,
such as barplots, boxplots, and scatter plots, to analyze the distribution and
characteristics of data related to 'Type', 'Vendor', 'Process Size', 'Vendor Distribution',
and 'TDP'.


Author: Zhengnan Li
Date: December 6, 2023
"""

from .data_summary import DataProcess
from .exploratory_data_analysis import EDA
from .inferences import Inferences


def main():
    """
    Run Chip analysis as a script.
    """
    print('------------------------------------------------')
    print('Chip-analysis-and-prediction')
    print('------------------------------------------------')
    print('Data Summary')
    data_summary()
    print('------------------------------------------------')
    print('EDA')
    exploratory_data_analysis()
    print('------------------------------------------------')
    print('Inferences')
    inferences_prediction()
    print('------------------------------------------------')


def data_summary():
    """
    Perform data summary for chip analysis.

    Returns:
        DataProcess: An instance of the DataProcess class.
    """
    my_file_path = 'https://raw.githubusercontent.com/Zhengnan817/Project-3-Data-Reconstruction-and-Analysis/main/src/chip_analysis/data/chip_dataset.csv'
    cpu_table = DataProcess(my_file_path)
    df = cpu_table.view_data()
    print(df.head())

    column_types = df.dtypes
    print(column_types)

    return cpu_table


def exploratory_data_analysis():
    """
    Perform exploratory data analysis for chip analysis.
    """
    my_file_path = 'https://raw.githubusercontent.com/Zhengnan817/Project-3-Data-Reconstruction-and-Analysis/main/src/chip_analysis/data/chip_dataset.csv'
    cpu_table = DataProcess(my_file_path)
    df = cpu_table.view_data()
    EDA_part = EDA(df)

    EDA_part.vendor_distribution_sns()

    EDA_part.vendor_distribution_plt()

    EDA_part.type_vendor_sns()

    EDA_part.type_vendor_plt()

    EDA_part.process_size_sns()

    EDA_part.process_size_plt()

    EDA_part.process_size_vendor_sns()

    EDA_part.process_size_vendor_plt()

    EDA_part.TDP_distribution_sns()

    EDA_part.TDP_distribution_plt()


def inferences_prediction():
    """
    Perform inferences for chip analysis.
    """
    print('This part include EDA and Predictive model.')
    my_file_path = 'https://raw.githubusercontent.com/Zhengnan817/Project-3-Data-Reconstruction-and-Analysis/main/src/chip_analysis/data/chip_dataset.csv'
    cpu_table = DataProcess(my_file_path)
    df = cpu_table.view_data()
    inferences_analysis = Inferences(df)
    inferences_analysis.vendor_type_plt()

    inferences_analysis.vendor_type_sns()

    inferences_analysis.chip_attribute()

    inferences_analysis.ave_freq_type_ven()

    inferences_analysis.freq_and_TDP()

    inferences_analysis.chip_attribute_overview()

    inferences_analysis.freq_trend()

    inferences_analysis.cpu_freq_and_attri()

    inferences_analysis.gpu_freq_and_attri()


main()
