import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda(file_path='data/loan_data.csv'):
    df = pd.read_csv(file_path)
    os.makedirs('notebooks/plots', exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Default', data=df)
    plt.title('Distribution of Loan Defaults')
    plt.savefig('notebooks/plots/target_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('notebooks/plots/correlation_heatmap.png')
    plt.close()
    
    # 3. Distribution Plots for key features
    features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'DTI']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('notebooks/plots/feature_distributions.png')
    plt.close()
    
    # 4. Boxplots for Outlier Detection
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig('notebooks/plots/feature_boxplots.png')
    plt.close()
    
    print("EDA plots generated and saved in notebooks/plots/")

if __name__ == "__main__":
    perform_eda()
