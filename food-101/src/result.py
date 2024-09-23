"""
This script reads accuracy or loss data from multiple CSV files and generates a plot to compare the performance of different models over epochs.

It loads CSV files that contain epoch-wise metrics (accuracy or loss), and then it plots the selected metric for each model. The resulting plot is saved as a PNG file.

Modules:
    - pandas: Used to load and manipulate the CSV data.
    - matplotlib: Used to generate the plots.
    - config: Contains configuration settings, such as paths for saving the plot.

Usage:
    - The function `plot_accuracy` is the main function that takes in a list of CSV files, model names, the y-axis feature label (e.g., accuracy or loss), and a human-readable feature name for labeling the plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from config import Config


def plot_accuracy(files, model_names, y_feature_label, y_feature_name):
    """
    Plots the given feature (accuracy or loss) over epochs for multiple models.

    This function reads CSV files that contain model performance metrics for each epoch and plots the selected feature (accuracy, loss, etc.) against epochs for each model. The plot is then saved as a PNG file in the directory specified by `Config.PLOTS_DIR`.

    Args:
        files (list of str): List of file paths to the CSV files, where each CSV contains epoch-wise data for one model.
        model_names (list of str): List of model names, corresponding to the CSV files, used for labeling the plot.
        y_feature_label (str): The name of the column in the CSV file to be plotted on the y-axis (e.g., "accuracy", "loss").
        y_feature_name (str): A human-readable name for the y-axis feature (e.g., "Accuracy", "Loss"), used in the plot title and labels.

    Behavior:
        - Reads the CSV files.
        - Plots the specified y-axis feature against the epochs for each model.
        - Saves the resulting plot as a PNG file named after the y_feature_label in the directory specified by `Config.PLOTS_DIR`.

    Example:
        plot_accuracy(["model1.csv", "model2.csv"], ["Model 1", "Model 2"], "accuracy", "Accuracy")

    Output:
        - A plot comparing the specified feature across different models is saved as a PNG file in the configured directory.
    """
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Plot the accuracy vs epoch for each model
    plt.figure(figsize=(10, 6))

    for i, df in enumerate(dfs):
        plt.plot(
            df["epoch"],
            df[y_feature_label],
            label=f"Model {model_names[i]} {y_feature_name}",
        )

    plt.title(f"Epoch vs {y_feature_name} for 4 Models")
    plt.xlabel("Epoch")
    plt.ylabel(y_feature_name)
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.savefig(f"{Config.PLOTS_DIR}{y_feature_label}.png")
