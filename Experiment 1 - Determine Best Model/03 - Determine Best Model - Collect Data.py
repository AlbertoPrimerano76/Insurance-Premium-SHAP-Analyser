"""
Insurance Model Evaluation Script

This script evaluates multiple machine learning models across different insurance domains,
comparing their performance and analyzing feature importance using SHAP values.

Usage:
    python insurance_model_evaluation.py

Note: Ensure all required libraries are installed and experiment-specific modules
      are available in the same directory or in the Python path.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from tqdm.auto import tqdm
from collections import defaultdict
from utilities import pre_process_data
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_output_filename() -> str:
    """Prompt user for output filename and return it with .csv extension."""
    return input("Please provide the output file name (without extension): ") + '.csv'

# Define the models to be evaluated
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None, random_state=42)
}

# Define experiments and their specific important features
experiments = [
    {
        'name': 'Auto Premium',
        'module': 'auto_insurance_premium',
        'prefix': 'auto',
        'important_features': {'Age', 'Gender', 'State', 'Business Use', 'Annual kilometers'}
    },
    {
        'name': 'Cyber Security',
        'prefix': 'cyber',
        'module': 'cybersecurity_insurance_premium',
        'important_features': {'Company Size', 'Industry Risk', 'Security Score', 'Data Sensitivity', 
                               'Business Interruption Cost'}
    },
    {
        'name': 'Environment Liability',
        'prefix': 'env_liab',
        'module': 'env_liability_insurance_premium',
        'important_features': {'Industry Type', 'Company Size', 'Pollution Risk', 
                               'Regulatory Compliance', 'Years of Operation', 
                               'Incident History', 'Coverage Limit'}
    }
]

# Constants
TARGET_COLUMN = 'Premium'
N_ITERATIONS = 1000
SAMPLE_SIZES = [100, 500, 1000, 5000]
EXPERIMENTS_TO_RUN = ['Auto Premium', 'Cyber Security', 'Environment Liability']

def run_experiment(experiment, output_file):
    """Run a single experiment and save results to the specified output file."""
    experiment_name = experiment['name']
    experiment_module_name = experiment['module']
    experiment_prefix = experiment['prefix']
    important_features = experiment['important_features']

    try:
        # Dynamically import the module containing the `generate_test_data` method
        experiment_module = importlib.import_module(experiment_module_name)
        generate_test_data = getattr(experiment_module, 'generate_test_data')
    except ImportError as e:
        logger.error(f"Failed to import module {experiment_module_name}: {e}")
        return

    results_table = []
    exp_index = 1

    # Set up the progress bar
    total_iterations = N_ITERATIONS * len(SAMPLE_SIZES) * len(models)
    with tqdm(total=total_iterations, desc=f"Running {experiment_name}", position=0, leave=True) as pbar:
        for iteration in range(N_ITERATIONS):
            for size in SAMPLE_SIZES:
                try:
                    # Generate and preprocess data
                    data = generate_test_data(size)
                    X_processed, y, _ = pre_process_data(data, target_column=TARGET_COLUMN)
                    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=7)

                    for name, model in models.items():
                        try:
                            # Train and evaluate model
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Calculate metrics
                            r2_score = model.score(X_test, y_test)
                            mse_score = mean_squared_error(y_test, y_pred)
                            mae_score = mean_absolute_error(y_test, y_pred)
                            rmse_score = np.sqrt(mse_score)
                            accuracy_score = explained_variance_score(y_test, y_pred)

                            # Perform SHAP analysis
                            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, CatBoostRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor)):
                                explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                                shap_values = explainer.shap_values(X_test, check_additivity=False)
                            else:
                                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
                                shap_values = explainer.shap_values(X_test)

                            # Calculate SHAP feature importance and determine top features
                            shap_feature_importance = dict(zip(X_processed.columns, np.mean(np.abs(shap_values), axis=0)))
                            sorted_features = sorted(shap_feature_importance.items(), key=lambda item: item[1], reverse=True)
                            top_features = set([feature for feature, _ in sorted_features[:len(important_features)]])
                            top_shap_features = dict(sorted_features[:len(important_features)])

                            # Calculate match percentage
                            match_percentage = len(important_features.intersection(top_features)) / len(important_features) * 100.

                            # Store results
                            results_table.append({
                                'Experiment Index': exp_index,
                                'Model': name,
                                'Sample Size': size,
                                'R²': r2_score,
                                'MAE': mae_score,
                                'RMSE': rmse_score,
                                'Accuracy': accuracy_score,
                                'SHAP Match Percentage': match_percentage,
                                'Top SHAP Features': top_shap_features
                            })

                            exp_index += 1
                            pbar.update(1)
                            pbar.set_postfix(iteration=iteration+1, size=size, model=name)
                        except Exception as e:
                            logger.error(f"Error with model {name} at sample size {size}: {e}")
                            pbar.update(1)
                except Exception as e:
                    logger.error(f"Error during processing at sample size {size}: {e}")
                    pbar.update(len(models))

    # Save results to CSV
    raw_results_df = pd.DataFrame(results_table)
    raw_results_df.to_csv(output_file, index=False)
    logger.info(f"Results for {experiment_name} saved to {output_file}")

def main():
    output_file = get_output_filename()
    for experiment in experiments:
        if experiment['name'] in EXPERIMENTS_TO_RUN:
            run_experiment(experiment, output_file)

if __name__ == "__main__":
    main()