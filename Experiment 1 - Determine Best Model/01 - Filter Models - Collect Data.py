"""
Insurance Model Evaluation Script

This script evaluates multiple machine learning models across different insurance domains,
comparing their performance in identifying important features using SHAP values.

Key components:
1. Model definitions
2. Experiment configurations
3. Data generation and preprocessing
4. Model training and evaluation
5. SHAP analysis
6. Results logging and export

Usage:
    python insurance_model_evaluation.py

Note: Ensure all required libraries are installed and experiment-specific modules
      (auto_insurance_premium, cybersecurity_insurance_premium, env_liability_insurance_premium)
      are available in the same directory or in the Python path.
"""

import importlib
import shap
import traceback
import logging
import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Import utilities from a different path
sys.path.append(os.path.abspath('../Calculations'))
from utilities import pre_process_data 

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

from typing import Dict, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the models
MODELS: Dict[str, object] = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0),
    'ElasticNet': ElasticNet(random_state=42),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None, random_state=42)
}

# Define experiments and their specific important features
EXPERIMENTS: List[Dict[str, Union[str, Set[str]]]] = [
    {
        'name': 'Auto Premium',
        'module': 'auto_insurance_premium',
        'prefix': 'auto',
        'important_features': {'Age', 'Gender', 'State', 'Business Use', 'Annual kilometers'}
    },
    {
        'name': 'Cyber Security',
        'module': 'cybersecurity_insurance_premium',
        'prefix': 'cyber',
        'important_features': {'Company Size', 'Industry Risk', 'Security Score', 'Data Sensitivity', 
                               'Business Interruption Cost'}
    },
    {
        'name': 'Environment Liability',
        'module': 'env_liability_insurance_premium',
        'prefix': 'env_liab',
        'important_features': {'Industry Type', 'Company Size', 'Pollution Risk', 
                               'Regulatory Compliance', 'Years of Operation', 
                               'Incident History', 'Coverage Limit'}
    }
]

# Constants
TARGET_COLUMN = 'Premium'
N_ITERATIONS = 500
SAMPLE_SIZES = [100, 500, 1000, 2000]
EXPERIMENTS_TO_RUN = ['Auto Premium', 'Cyber Security', 'Environment Liability']
TREE_BASED_MODELS = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'CatBoost', 'ExtraTrees', 'HistGradientBoosting']

def get_output_filename() -> str:
    """Prompt user for output filename and return it with .csv extension."""
    return input("Please provide the output file name (without extension): ") + '.csv'

def import_experiment_module(module_name: str):
    """Dynamically import the module containing the `generate_test_data` method."""
    try:
        experiment_module = importlib.import_module(module_name)
        return getattr(experiment_module, 'generate_test_data')
    except Exception as e:
        logger.error(f"Error importing module {module_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def process_data(generate_test_data, sample_size: int):
    """Generate and preprocess data for the experiment."""
    try:
        data = generate_test_data(sample_size)
        X_processed, y, _ = pre_process_data(data, target_column=TARGET_COLUMN)
        return train_test_split(X_processed, y, test_size=0.2, random_state=42)
    except Exception as e:
        logger.error(f"Error processing data with sample size {sample_size}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def train_and_evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test, important_features: Set[str]):
    """Train the model, perform SHAP analysis, and evaluate feature importance."""
    try:
        model.fit(X_train, y_train)
        
        if model_name in TREE_BASED_MODELS:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train[:10])
            shap_values = explainer.shap_values(X_test)
        
        shap_feature_importance = dict(zip(X_train.columns, np.mean(np.abs(shap_values), axis=0)))
        sorted_features = sorted(shap_feature_importance.items(), key=lambda item: item[1], reverse=True)
        top_features = set([feature for feature, _ in sorted_features[:len(important_features)]])
        match_percentage = len(important_features.intersection(top_features)) / len(important_features) * 100
        top_features_string = ", ".join([f"{feature}: {importance:.4f}" for feature, importance in sorted_features[:len(important_features)]])
        
        return match_percentage, top_features_string
    except Exception as e:
        logger.error(f"Error training model {model_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None, None

def main():
    output_file_name = get_output_filename()
    results = []
    total_iterations = len(EXPERIMENTS_TO_RUN) * len(SAMPLE_SIZES) * len(MODELS) * N_ITERATIONS

    with tqdm(total=total_iterations, desc="Total Progress", unit="iteration") as pbar:
        for experiment in EXPERIMENTS:
            if experiment['name'] not in EXPERIMENTS_TO_RUN:
                continue

            generate_test_data = import_experiment_module(experiment['module'])
            if not generate_test_data:
                pbar.update(len(SAMPLE_SIZES) * len(MODELS) * N_ITERATIONS)
                continue

            for sample_size in SAMPLE_SIZES:
                logger.info(f"Processing {experiment['name']} with sample size {sample_size}")
                
                for _ in range(N_ITERATIONS):
                    data = process_data(generate_test_data, sample_size)
                    if not data:
                        pbar.update(len(MODELS))
                        continue

                    X_train, X_test, y_train, y_test = data

                    for model_name, model in MODELS.items():
                        match_percentage, top_features = train_and_evaluate_model(
                            model_name, model, X_train, X_test, y_train, y_test, experiment['important_features']
                        )
                        
                        if match_percentage is not None:
                            results.append({
                                'Experiment': experiment['name'],
                                'Model': model_name,
                                'Sample Size': sample_size,
                                'SHAP Match %': match_percentage,
                                'Top Features': top_features
                            })
                            logger.info(f"{model_name} - Correct Features Identified: {match_percentage:.2f}%")
                        
                        pbar.update(1)

    df_results = pd.DataFrame(results)
    try:
        df_results.to_csv(output_file_name, index=False)
        logger.info(f"Results saved to {output_file_name}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()