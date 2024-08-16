import os
import random
import pandas as pd
from datetime import datetime
from pycaret.regression import setup, compare_models, pull, tune_model,ensemble_model,finalize_model,save_model

def pycaret_executor(sample_data, target_column, categorical_features=[], test_size=0.2):
    """
    Executes the PyCaret setup and model comparison.

    Parameters:
    sample_data (DataFrame): The dataset to be used.
    target_column (str): The name of the target column in the dataset.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    tuple: A tuple containing the session ID and the best model.
    """
    session_id = random.randint(1, 10000)
    print(f"Processing: Sample Size = {len(sample_data)}, Session ID = {session_id}")
    exp = setup(
        data=sample_data,
        target=target_column,
        session_id=session_id,
        train_size=1-test_size,
        normalize=True,
        verbose=False
    )
    best_model = compare_models()
    return best_model

def analyze_model_performances(sample_data, target_column, output_file, categorical_features=[], test_size=0.2):
    """
    Analyzes the performance of models using the PyCaret library and saves the results.

    Parameters:
    sample_data (DataFrame): The dataset to be used.
    target_column (str): The name of the target column in the dataset.
    output_file (str): Path to the CSV file to save the results.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    best_model: The best model according to the comparison.
    """
    best_model = pycaret_executor(sample_data, target_column,categorical_features, test_size)
    metrics = pull()
    results = []
    if not metrics.empty:
        best_r2_model = metrics.loc[metrics['R2'].idxmax(), 'Model']
        best_r2_value = metrics['R2'].max()

        best_mse_model = metrics.loc[metrics['MSE'].idxmin(), 'Model']
        best_mse_value = metrics['MSE'].min()

        best_mae_model = metrics.loc[metrics['MAE'].idxmin(), 'Model']
        best_mae_value = metrics['MAE'].min()

        results.append({
            'Sample Size': len(sample_data),
            'Best R2 Model': best_r2_model,
            'Best R2 Value': best_r2_value,
            'Best MSE Model': best_mse_model,
            'Best MSE Value': best_mse_value,
            'Best MAE Model': best_mae_model,
            'Best MAE Value': best_mae_value
        })
    else:
        results.append({
            'Sample Size': len(sample_data),
            'Best R2 Model': None,
            'Best R2 Value': 0,
            'Best MSE Model': None,
            'Best MSE Value': 0,
            'Best MAE Model': None,
            'Best MAE Value': 0
        })

    results_df = pd.DataFrame(results)
    # Check if the file exists
    if os.path.isfile(output_file):
        # Append without header
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        results_df.to_csv(output_file, mode='w', header=True, index=False)
    return best_model
