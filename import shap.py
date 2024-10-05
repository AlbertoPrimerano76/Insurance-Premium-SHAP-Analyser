import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from tqdm import tqdm
from collections import defaultdict
from utilities import pre_process_data
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, HistGradientBoostingRegressor

models = {
#    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1),
#    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0),
#    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None, random_state=42)
}

# Define experiments and their specific important features
experiments = [
    {
        'name': 'Auto Premium',
        'module' : 'auto_insurance_premium',
        'prefix': 'auto',
        'important_features': {'Age', 'Gender', 'State', 'Business Use', 'Annual kilometers'}
    },
    {
        'name': 'Cyber Security',
        'prefix': 'cyber',
        'module' : 'cybersecurity_insurance_premium',
        'important_features': {'Company Size', 'Industry Risk', 'Security Score', 'Data Sensitivity', 
                                       'Business Interruption Cost'}
    },
    {
        'name': 'Environment Liability',
        'prefix': 'env_liab',
        'module' : 'env_liability_insurance_premium',
        'important_features': {'Industry Type', 'Company Size', 'Pollution Risk', 
                                       'Regulatory Compliance', 'Years of Operation', 
                                       'Incident History', 'Coverage Limit'}
    }
]

# Iteration 1
# Number of times to run the tests
#N = 1000
# Sample sizes
#sample_sizes = [100,500,1000,5000]

# 1 -> 2 Reduced number of iterations, reduced number of samples type, increased samples size

# Iteration 2
# Number of times to run the tests
N = 1000
# Sample sizes
sample_sizes = [100,1000,10000,100000]



# Array to hold the experiment names you want to run
#experiments_to_run = ['Auto Premium', 'Cyber Security', 'Environment Liability']
experiments_to_run = ['Environment Liability']


# Define Target Column
target_column = 'Premium'

# Main loop to iterate through experiments
for experiment in experiments:
    if experiment['name'] in experiments_to_run:
        experiment_name = experiment['name']
        experiment_module_name = experiment['module']
        experiment_prefix = experiment['prefix']
        important_features = experiment['important_features']

        # Dynamically import the module containing the `generate_test_data` method
        experiment_module = importlib.import_module(experiment_module_name)
        generate_test_data = getattr(experiment_module, 'generate_test_data')

        # Initialize dictionaries to collect results across multiple runs
        overall_results = defaultdict(lambda: defaultdict(lambda: {'matches': 0, 'R²': [], 'MAE': [], 'RMSE': [], 'Accuracy': []}))

        # Table to collect results per experiment
        results_table = []
        exp_index = 1

        # Set up the progress bar for N * sample_sizes total iterations
        total_iterations = N * len(sample_sizes) * len(models)
        with tqdm(total=total_iterations, desc=f"Running {experiment_name}", leave=True) as pbar:
            for iteration in range(N):
                for size in sample_sizes:
                    try:
                        # Generate dataset using the experiment-specific method
                        data = generate_test_data(size)

                        X_processed, y, mappings = pre_process_data(data, target_column=target_column)

                        # Split the data into training and test sets
                        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=7)

                        # Evaluate each model
                        for name, model in models.items():
                            try:
                                # Fit the model on the training data
                                model.fit(X_train, y_train)

                                # Evaluate the model on the test data
                                y_pred = model.predict(X_test)

                                # Calculate evaluation metrics
                                r2_score = model.score(X_test, y_test)
                                mse_score = mean_squared_error(y_test, y_pred)
                                mae_score = mean_absolute_error(y_test, y_pred)
                                rmse_score = np.sqrt(mse_score)
                                accuracy_score = explained_variance_score(y_test, y_pred)

                                # SHAP analysis for tree-based models
                                explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                                shap_values = explainer.shap_values(X_test,check_additivity=False)

                                # Calculate SHAP feature importance
                                shap_feature_importance = dict(zip(X_processed.columns, np.mean(np.abs(shap_values), axis=0)))

                                # Determine the top SHAP features
                                sorted_features = sorted(shap_feature_importance.items(), key=lambda item: item[1], reverse=True)
                                sorted_feature_names = [feature for feature, importance in sorted_features]
                                top_features = set(sorted_feature_names[:len(important_features)])

                                # Store the top features and their relevance
                                top_shap_features = {feature: importance for feature, importance in sorted_features[:len(important_features)]}

                                # Calculate the percentage of important features present in SHAP
                                match_percentage = len(important_features.intersection(top_features)) / len(important_features) * 100

                                # Store all the results in a row
                                results_table.append({
                                    'Experiment Index': exp_index,
                                    'Model': name,
                                    'Sample Size': size,
                                    'R²': r2_score,
                                    'MAE': mae_score,
                                    'RMSE': rmse_score,
                                    'Accuracy': accuracy_score,
                                    'SHAP Match Percentage': match_percentage,
                                    'Top SHAP Features': top_shap_features  # Include top SHAP features and their importance
                                })

                                exp_index += 1
                                # Advance the outer progress bar
                                pbar.update(1)
                            except Exception as e:
                                tqdm.write(f"Error with model {name} at sample size {size}: {e}")
                                continue  # Skip to the next model

                    except Exception as e:
                        tqdm.write(f"Error during processing at sample size {size}: {e}")
                        continue  # Skip to the next sample size

               

        # Convert results table to DataFrame for better analysis
        raw_results_df = pd.DataFrame(results_table)

        # Save the raw results for this experiment to a CSV file
        csv_filename = f"raw_results_{experiment_prefix}.csv"
        raw_results_df.to_csv(csv_filename, index=False)

        print(f"Results for {experiment_name} saved to {csv_filename}")
