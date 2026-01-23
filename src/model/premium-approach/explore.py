# Install required packages if not already installed
# !pip install seaborn

## Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from IPython.display import display
import seaborn as sns
from datasets import insurance_new


## Exploratory Data Analysis function for insurance dataset
def explore_insurance_data(df):
    
    print("Exploratory Data Analysis - Insurance Dataset\n")
    print("-" * 50)
    
    ## Explore dataset shape
    print("#" * 50)
    print(f"Dataset Shape: {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    
    # Display first 5 rows of the dataset
    print("#" * 50)
    print("First 5 rows of the dataset:")
    display(df.head())
    
    # Display dataset info
    print("#" * 50)
    print("\nDataset Info:")
    df.info()
    
    # Display statistical summary
    print("#" * 50)
    print("\nStatistical Summary:")
    display(df.describe())
    
    # Display duplicated rows information
    print("#" * 50)
    print("\nDuplicated Rows:")
    print(f"The count of duplicated rows is {df.duplicated().sum():,}")
    
    # Display missing values information
    print("#" * 50) 
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("#" * 50)
    ## Visualize missing values using a heatmap
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values Heatmap - Insurance Data")
    plt.show()
    
    
    
    print("#" * 50)
    # Visualize categorical feature distributions with percentage labels
    print("\nCategorical Feature Distribution:") ## include percentage distribution
    categorical_cols = df.select_dtypes(include=['object']).columns  
    
    ## Visualize categorical feature distributions with percentage labels
    for col in categorical_cols:
        plt.figure(figsize=(6, 5))
        ax = sns.countplot(
            data=df,
            x=col,
            order=df[col].value_counts().index,
            stat="percent"
        )

        plt.title(f'Distribution of {col} in %')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)

        # add percentage labels on bars
        for p in ax.patches:
            value = p.get_height()
            ax.annotate(
                f"{value:.1f}%",
                (p.get_x() + p.get_width() / 2, value),
                ha='center',
                va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )

        plt.show()

    print("#" * 50)
        # Visualize numerical feature distributions     
    print("\nNumerical Feature Distribution:")
    numerical_cols = df.select_dtypes(exclude=['object', 'datetime']).columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
        

if __name__ == "__main__":
    # Execute exploratory analysis when the script is run directly
    print("Starting Exploratory Data Analysis...\n")
    explore_insurance_data(insurance_new)
    print("\nExploratory Data Analysis Completed.")