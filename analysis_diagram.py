import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pycaret.classification import setup, compare_models, interpret_model, plot_model



def plot_model_performance(input_csv, sample_sizes, output_plots_prefix):
    """
    Plots the model performance metrics (R2, MSE, MAE) against sample sizes.

    Parameters:
    input_csv (str): Path to the CSV file containing the performance results.
    sample_sizes (list): List of sample sizes used in the analysis.
    output_plots_prefix (str): Prefix for the output plot image files.
    """
    # Load the dataframe
    df = pd.read_csv(input_csv)

    # Calculate variance and mean for MSE and MAE thresholds
    variance = df['Best MSE Value'].max() / 0.1  # Reverse the 10% rule for MSE threshold
    mean = df['Best MAE Value'].max() / 0.1      # Reverse the 10% rule for MAE threshold

    # Define thresholds
    r2_threshold = 0.8
    mse_threshold = 0.1 * variance
    mae_threshold = 0.1 * mean

    # Define the logarithmic function for fitting
    def fit_func(x, a, b, c):
        return a * np.log(b * x) + c

    # Fit models for the dataframe
    params_r2, _ = curve_fit(fit_func, df['Sample Size'], df['Best R2 Value'])
    params_mse, _ = curve_fit(fit_func, df['Sample Size'], df['Best MSE Value'])
    params_mae, _ = curve_fit(fit_func, df['Sample Size'], df['Best MAE Value'])

    # Predict sample size for each threshold
    def find_sample_size(params, threshold, greater_than=True):
        for sample_size in sample_sizes:
            if greater_than and fit_func(sample_size, *params) >= threshold:
                return sample_size
            elif not greater_than and fit_func(sample_size, *params) <= threshold:
                return sample_size
        return None

    # Predict minimal sample sizes for each metric
    min_samples_r2 = find_sample_size(params_r2, r2_threshold)
    min_samples_mse = find_sample_size(params_mse, mse_threshold, greater_than=False)
    min_samples_mae = find_sample_size(params_mae, mae_threshold, greater_than=False)

    # Ensure all values are valid integers
    min_samples_needed = max(
        min_samples_r2 if min_samples_r2 is not None else float('inf'),
        min_samples_mse if min_samples_mse is not None else float('inf'),
        min_samples_mae if min_samples_mae is not None else float('inf')
    )

    # Convert sample sizes to a NumPy array
    sample_sizes_array = np.array(sample_sizes)

    # Calculate fitted values
    r2_values = fit_func(sample_sizes_array, *params_r2)
    mse_values = fit_func(sample_sizes_array, *params_mse)
    mae_values = fit_func(sample_sizes_array, *params_mae)

    # Create a figure with three subplots
    plt.figure(figsize=(18, 6))

    # R2 Score plot
    plt.subplot(1, 3, 1)
    plt.plot(sample_sizes_array, r2_values, label='R2', color='blue')
    plt.axhline(y=r2_threshold, color='r', linestyle='--', label='Threshold')
    if min_samples_r2 is not None:
        plt.scatter(min_samples_r2, r2_threshold, color='blue')
    plt.xlabel('Sample Size')
    plt.ylabel('R2 Value')
    plt.title('R2 Value vs Sample Size')
    plt.legend()

    # MSE plot
    plt.subplot(1, 3, 2)
    plt.plot(sample_sizes_array, mse_values, label='MSE', color='green')
    plt.axhline(y=mse_threshold, color='r', linestyle='--', label='Threshold')
    if min_samples_mse is not None:
        plt.scatter(min_samples_mse, mse_threshold, color='green')
    plt.xlabel('Sample Size')
    plt.ylabel('MSE Value')
    plt.title('MSE Value vs Sample Size')
    plt.legend()

    # MAE plot
    plt.subplot(1, 3, 3)
    plt.plot(sample_sizes_array, mae_values, label='MAE', color='purple')
    plt.axhline(y=mae_threshold, color='r', linestyle='--', label='Threshold')
    if min_samples_mae is not None:
        plt.scatter(min_samples_mae, mae_threshold, color='purple')
    plt.xlabel('Sample Size')
    plt.ylabel('MAE Value')
    plt.title('MAE Value vs Sample Size')
    plt.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{output_plots_prefix}_plots.png")

    # Display minimum sample sizes needed
    print(f"Minimum sample size for R2 threshold: {min_samples_r2}")
    print(f"Minimum sample size for MSE threshold: {min_samples_mse}")
    print(f"Minimum sample size for MAE threshold: {min_samples_mae}")
    print(f"Overall minimum sample size needed: {min_samples_needed}")


def interpret_xgboost_model(best_model):
    """
    Interpret the XGBoost model using SHAP, feature importance, and PDP.

    Parameters:
    best_model: The trained XGBoost model.
    """
    # SHAP Summary Plot
    interpret_model(best_model, plot='summary')
    
    # Feature Importance Plot
    plot_model(best_model, plot='feature')
    
    # Partial Dependence Plot for a specific feature (e.g., 'feature_name')
    plot_model(best_model, plot='pdp', feature='feature_name')
