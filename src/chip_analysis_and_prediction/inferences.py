# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Inferences:
    """
    A class for generating various visualizations and inferences from Chips of GPU and CPU.

    This class provides methods to visualize and analyze data, specifically tailored for a DataFrame
    containing information about vendors, types, process sizes, release dates, and attributes of chips.

    Attributes:
    - df (pd.DataFrame): The input DataFrame for analysis.

    Methods:
    - __init__(self, df): Initialize the Inferences class with a DataFrame.
    - vendor_type_sns(self): Generate a Seaborn bar plot to visualize the distribution of Vendor types.
    - vendor_type_plt(self): Generate a Matplotlib bar plot to visualize the distribution of Vendor types.
    - ave_freq_type_ven(self): Generate a Seaborn bar plot to visualize the average frequency for each product and vendor.
    - chip_attribute(self): Generate a Seaborn box plot to visualize chip attributes across vendors.
    - freq_and_TDP(self): Generate a Matplotlib bar plot to visualize the average TDP by vendor.
    - process_size_date_analysis(self): Perform analysis on the relationship between process size and release date.
    - chip_attribute_overview(self): Generate a Seaborn pairplot for numeric features in the chip dataset.
    - freq_trend(self): Analyze the frequency trend over time using a scatter plot and regression line.
    - cpu_freq_and_attri(self): Train a RandomForestRegressor model to predict CPU frequency and evaluate its performance.
    - gpu_freq_and_attri(self): Train a RandomForestRegressor model to predict GPU frequency and evaluate its performance.
    """

    def __init__(self, df):
        """
        Initialize the Inferences class with a DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame for analysis.
        """
        self.df = df

    def vendor_type_sns(self):
        """
        Generate a Seaborn bar plot to visualize the distribution of Vendor types.

        The plot shows the count of each type for each vendor.

        Returns:
        None
        """
        # Create a cross-tabulation and plot a bar chart using Seaborn
        crosstab = pd.crosstab(index=self.df['Vendor'], columns=self.df['Type'])
        crosstab.plot.bar(figsize=(7, 4), rot=0).set(ylabel='Type', title='Vendor')

    def vendor_type_plt(self):
        """
        Generate a Matplotlib bar plot to visualize the distribution of Vendor types.

        The plot shows the count of each type for each vendor.

        Returns:
        None
        """
        # Create a cross-tabulation and plot a bar chart using Matplotlib
        crosstab = pd.crosstab(index=self.df['Vendor'], columns=self.df['Type'])
        plt.figure(figsize=(7, 4))
        for col in crosstab.columns:
            plt.bar(crosstab.index, crosstab[col], label=col)

        plt.xlabel('Vendor')
        plt.ylabel('Type')
        plt.title('Vendor Type Distribution')
        plt.legend()
        plt.show()

    def ave_freq_type_ven(self):
        """
        Generate a Seaborn bar plot to visualize the average frequency for each product and vendor.

        Returns:
        None
        """
        # Create a pivot table for average frequency by product and vendor
        pivot_data = self.df.pivot_table(
            index=['Type', 'Vendor'], values='Process Size', aggfunc='mean'
        ).reset_index()

        # Plot the average frequency for each product and vendor using Seaborn
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='Type',
            y='Process Size',
            hue='Vendor',
            data=pivot_data,
            palette='viridis',
        )
        plt.title('Average Frequency for Each Vendor and Vendor')
        plt.xlabel('Type')
        plt.ylabel('Average Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Vendor', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def chip_attribute(self):
        """
        Generate a Seaborn box plot to visualize chip attributes across vendors.

        The plot shows the distribution of attributes (excluding Transistors and Freq) for each vendor.

        Returns:
        None
        """
        # Reshape the DataFrame for box plot and plot using Seaborn
        re_release_vendor = pd.melt(
            self.df.drop(['Transistors', 'Freq'], axis=1),
            id_vars=['Product', 'Type', 'Release Date', 'Vendor'],
            var_name='Attribute',
            value_name='Value',
        )
        plt.figure(figsize=(8, 8))
        sns.boxplot(x='Attribute', y='Value', data=re_release_vendor, hue='Vendor')
        plt.title('Attribute View')
        plt.show()

    def freq_and_TDP(self):
        """
        Generate a Matplotlib bar plot to visualize the average TDP by vendor.

        Returns:
        None
        """
        # Group data by vendor and plot the average Die Size using Matplotlib
        grouped_data = self.df.groupby('Vendor').agg({'Die Size': 'mean'}).reset_index()
        grouped_data.plot(kind='bar', y='Die Size', x='Vendor', legend=False)
        plt.title('Average Die Size by Vendor')
        plt.xlabel('Vendor')
        plt.ylabel('Average Die Size')
        plt.show()

    def process_size_date_analysis(self):
        """
        Perform analysis on the relationship between process size and release date.

        This method fits a linear regression model and displays the regression summary along with a scatter plot.

        Returns:
        None
        """
        # Prepare data for linear regression and plot the results using Seaborn and Matplotlib
        X = sm.add_constant(self.df['Release Date'].astype(int))
        y = self.df['Process Size']

        model = sm.OLS(y, X)
        results = model.fit()

        # Display regression summary
        print(results.summary())

        # Plotting the scatter plot
        sns.scatterplot(data=self.df, x='Release Date', y='Process Size', hue='Type')
        plt.title('Scatter Plot with Regression Line')

        # Plotting the regression line
        plt.plot(self.df['Release Date'], results.predict(X), color='red', linewidth=2)

        plt.show()

    def chip_attribute_overview(self):
        """
        Generate a Seaborn pairplot for numeric features in the chip dataset.

        Returns:
        None
        """
        # Create a pairplot for numeric features using Seaborn
        sns.pairplot(self.df, hue='Type', diag_kind='kde')
        plt.suptitle('Pairplot for Numeric Features', y=1.02)
        plt.show()

    def freq_trend(self):
        """
        Analyze the frequency trend over time using a scatter plot and regression line.

        Returns:
        None
        """
        # Convert 'Release Date' to datetime, perform linear regression, and plot using Seaborn and Matplotlib
        self.df['Release Date'] = pd.to_datetime(
            self.df['Release Date'], format='%m/%d/%y'
        )
        X = sm.add_constant(self.df['Release Date'].astype(int))
        y = self.df['Freq']

        model = sm.OLS(y, X)
        results = model.fit()

        # Display regression summary
        print(results.summary())

        # Plotting the scatter plot
        sns.scatterplot(data=self.df, x='Release Date', y='Freq', hue='Type')
        plt.title('Scatter Plot with Regression Line')

        # Plotting the regression line
        plt.plot(self.df['Release Date'], results.predict(X), color='red', linewidth=2)

        plt.show()

    def cpu_freq_and_attri(self):
        """
        Train a RandomForestRegressor model to predict CPU frequency and evaluate its performance.

        Returns:
        None
        """
        # Import the necessary data processing module
        from .data_summary import DataProcess

        # Load CPU data and preprocess it
        my_file_path = 'https://raw.githubusercontent.com/Zhengnan817/Project-3-Data-Reconstruction-and-Analysis/main/src/chip_analysis/data/chip_dataset.csv'
        cpu_table = DataProcess(my_file_path)
        df = cpu_table.view_data()
        columns_to_drop = ['Product', 'Vendor']
        df_new = df.drop(columns=columns_to_drop)
        cpu_data = df_new[df_new['Type'] == 'CPU']
        cpu_data = cpu_data.drop(columns=['Type'])

        # Separate features and target
        X = cpu_data[['Release Date', 'Process Size', 'Die Size', 'Transistors']]
        y = cpu_data['Freq']

        # Define preprocessor for numerical and categorical features
        numeric_features = ['Process Size', 'Die Size', 'Transistors']
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        categorical_features = ['Release Date']
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Define the model
        model = RandomForestRegressor()

        # Create the pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit the model on the training set
        pipeline.fit(X_train, y_train)

        # Evaluate the model on the testing set
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f'R-squared on Test Set: {r2}')
        y_pred = pipeline.predict(X_test)

        # Create a DataFrame with actual and predicted values
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        # Scatter plot using Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Actual', y='Predicted', data=result_df)
        plt.plot(
            result_df['Actual'],
            result_df['Actual'],
            color='red',
            linestyle='--',
            linewidth=2,
        )  # Standard line
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

    def gpu_freq_and_attri(self):
        """
        Train a RandomForestRegressor model to predict GPU frequency and evaluate its performance.

        Returns:
        None
        """
        # Import the necessary data processing module
        from .data_summary import DataProcess

        # Load GPU data and preprocess it
        my_file_path = 'https://raw.githubusercontent.com/Zhengnan817/Project-3-Data-Reconstruction-and-Analysis/main/src/chip_analysis/data/chip_dataset.csv'
        cpu_table = DataProcess(my_file_path)
        df = cpu_table.view_data()
        columns_to_drop = ['Product', 'Vendor']
        df_new = df.drop(columns=columns_to_drop)
        gpu_data = df_new[df_new['Type'] == 'GPU']
        gpu_data = gpu_data.drop(columns=['Type'])
        # Separate features and target
        X = gpu_data[['Release Date', 'Process Size', 'Die Size', 'Transistors']]
        y = gpu_data['Freq']

        # Define preprocessor for numerical and categorical features
        numeric_features = ['Process Size', 'Die Size', 'Transistors']
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        categorical_features = ['Release Date']
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        # Define the model
        model = RandomForestRegressor()

        # Create the pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit the model on the training set
        pipeline.fit(X_train, y_train)

        # Evaluate the model on the testing set
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f'R-squared on Test Set: {r2}')
        # Create a DataFrame with actual and predicted values
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        # Scatter plot using Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Actual', y='Predicted', data=result_df)
        plt.plot(
            result_df['Actual'],
            result_df['Actual'],
            color='red',
            linestyle='--',
            linewidth=2,
        )  # Standard line
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
