# This script generates and saves key visualizations from the raw dataset

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_utils import load_data

def _plot_target_distribution(df, output_dir):
    """Generates and saves the target variable distribution plot."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Variable (Heart Disease)')
    plt.xlabel('0 = No Heart Disease, 1 = Heart Disease')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    print("Generated: target_distribution.png")

def _plot_correlation_heatmap(df, output_dir):
    """Generates and saves the correlation heatmap of all features."""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Generated: correlation_heatmap.png")

def _plot_age_distribution(df, output_dir):
    """Generates and saves the age distribution plot."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True, bins=30)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()
    print("Generated: age_distribution.png")

def _plot_chest_pain_vs_target(df, output_dir):
    """Generates and saves the chest pain vs. target plot."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cp', hue='target', data=df)
    plt.title('Chest Pain Type vs. Heart Disease')
    plt.xlabel('Chest Pain Type (cp)')
    plt.ylabel('Count')
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'chest_pain_vs_target.png'))
    plt.close()
    print("Generated: chest_pain_vs_target.png")

def _plot_pairplot(df, output_dir):
    """Generates and saves the pairplot of key numerical features."""
    print("Generating pairplot... (this may take a moment)")
    pairplot_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
    pairplot = sns.pairplot(df[pairplot_features], hue='target', palette='viridis')
    pairplot.fig.suptitle('Pairwise Relationships of Key Features by Heart Disease', y=1.02)
    pairplot.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()
    print("Generated: pairplot.png")

def generate_visualizations():
    """
    Loads the raw data, creates several plots, and saves them to the
    'reports/figures' directory.
    """
    print("--- Starting Data Visualization ---")
    raw_data_path = '../data/raw/heart.csv'
    output_dir = '../reports/figures/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in: {output_dir}")

    # Load Raw Data
    try:
        df = load_data(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # Generate and Save all Visualizations
    _plot_target_distribution(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_age_distribution(df, output_dir)
    _plot_chest_pain_vs_target(df, output_dir)
    _plot_pairplot(df, output_dir)

    print("\n--- Data Visualization Complete ---")

if __name__ == "__main__":
    generate_visualizations()