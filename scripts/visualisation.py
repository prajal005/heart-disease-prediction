# This script generates and saves key visualizations from the raw dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_visualizations():
    """
    Loads the raw data, creates several plots, and saves them to the
    'reports/figures' directory.
    """
    print("--- Starting Data Visualization ---")

    # 1. Define File Paths

    raw_data_path = '../data/raw/heart.csv'
    output_dir = '../reports/figures/'

    # 2. Create Output Directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in: {output_dir}")

    # 3. Load Raw Data
    try:
        df = pd.read_csv('../data/raw/heart.csv')
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # 4. Generate and Save Visualizations

    # Plot 1: Target Variable Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Variable (Heart Disease)')
    plt.xlabel('0 = No Heart Disease, 1 = Heart Disease')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    print("Generated: target_distribution.png")

    # Plot 2: Correlation Heatmap of all features
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Generated: correlation_heatmap.png")

    # Plot 3: Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True, bins=30)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()
    print("Generated: age_distribution.png")

    # Plot 4: Chest Pain Type vs. Target
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cp', hue='target', data=df)
    plt.title('Chest Pain Type vs. Heart Disease')
    plt.xlabel('Chest Pain Type (cp)')
    plt.ylabel('Count')
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'chest_pain_vs_target.png'))
    plt.close()
    print("Generated: chest_pain_vs_target.png")

    # Plot 5: Pairplot of key numerical features
    print("Generating pairplot... (this may take a moment)")
    pairplot_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
    sns.pairplot(df[pairplot_features], hue='target', palette='viridis')
    plt.suptitle('Pairwise Relationships of Key Features by Heart Disease', y=1.02)
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()
    print("Generated: pairplot.png")


    print("\n--- Data Visualization Complete ---")

if __name__ == "__main__":
    generate_visualizations()
