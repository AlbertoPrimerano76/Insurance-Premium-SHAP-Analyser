import importlib
import shap
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utilities import pre_process_data
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import traceback

# Define the models
models = {
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
experiments = [
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

# Define Target Column
target_column = 'Premium'

# Sample sizes and repetitions
N = 50
sample_sizes = [100, 500, 1000, 2000]
experiments_to_run = ['Auto Premium', 'Cyber Security', 'Environment Liability']

# Get the output CSV file name as input
output_file_name = input("Please provide the output file name (without extension): ") + '.csv'

# Total iterations in tqdm: experiments * sample_sizes * number of models
total_iterations = len(experiments_to_run) * len(sample_sizes) * len(models) * N
results = []

# List of tree-based models that should use TreeExplainer
tree_based_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'CatBoost', 'ExtraTrees', 'HistGradientBoosting']

# Progress bar covering all iterations across experiments, sample sizes, and models
with tqdm(total=total_iterations, desc="Total Progress", unit="iteration") as pbar:

    for experiment in experiments:
        if experiment['name'] not in experiments_to_run:
            continue  # Skip experiments not in the list

        experiment_name = experiment['name']
        experiment_module_name = experiment['module']
        important_features = experiment['important_features']

        try:
            # Dynamically import the module containing the `generate_test_data` method
            experiment_module = importlib.import_module(experiment_module_name)
            generate_test_data = getattr(experiment_module, 'generate_test_data')
        except Exception as e:
            print(f"Error importing module {experiment_module_name}: {str(e)}")
            print(traceback.format_exc())
            pbar.update(len(sample_sizes) * len(models) * N)  # Skip this experiment in progress bar
            continue  # Skip to the next experiment

        # Iterate over different sample sizes
        for sample_size in sample_sizes:
            print(f"\nGenerating data with sample size {sample_size} for {experiment_name}...")

            # Repeat N times
            for repeat in range(N):
                try:
                    # Generate data (adjust size according to sample_size)
                    data = generate_test_data(sample_size)

                    # Pre-process the data
                    X_processed, y, mappings = pre_process_data(data, target_column=target_column)
                    feature_names = X_processed.columns  # Get feature names
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
                except Exception as e:
                    print(f"Error processing data for {experiment_name} with sample size {sample_size}: {str(e)}")
                    print(traceback.format_exc())
                    pbar.update(len(models))  # Skip this sample size for all models
                    continue  # Skip to the next sample size

                # Train each model and track the progress with tqdm
                print(f"Training models for {experiment_name} with sample size {sample_size}...")

                for model_name, model in models.items():
                    try:
                        # Fit the model
                        model.fit(X_train, y_train)

                        # Run SHAP analysis based on model type
                        if model_name in tree_based_models:
                            # Use TreeExplainer for tree-based models
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test)
                        else:
                            # Use KernelExplainer for other models like ElasticNet, MLPRegressor
                            explainer = shap.KernelExplainer(model.predict, X_train[:10])  # KernelExplainer can be slow, so sample training data
                            shap_values = explainer.shap_values(X_test)

                        # Calculate SHAP feature importance
                        shap_feature_importance = dict(zip(X_processed.columns, np.mean(np.abs(shap_values), axis=0)))
                        sorted_features = sorted(shap_feature_importance.items(), key=lambda item: item[1], reverse=True)
                        sorted_feature_names = [feature for feature, importance in sorted_features]
                        top_features = set(sorted_feature_names[:len(important_features)])
                        match_percentage = len(important_features.intersection(top_features)) / len(important_features) * 100
                        top_features_string = ", ".join([f"{feature}: {importance:.4f}" for feature, importance in sorted_features[:len(important_features)]])

                        # Append the result to the list
                        results.append({
                            'Experiment': experiment_name,
                            'Model': model_name,
                            'Sample Size': sample_size,
                            'SHAP Match %': match_percentage,
                            'Top Features': top_features_string
                        })

                        # Print the percentage of correctly identified features
                        print(f"{model_name} - Correct Features Identified: {match_percentage:.2f}%")

                    except Exception as e:
                        print(f"Error training model {model_name} for {experiment_name} with sample size {sample_size}: {str(e)}")
                        print(traceback.format_exc())

                    pbar.update(1)

                print(f"Finished training models for {experiment_name} with sample size {sample_size}.\n")

        print(f"Finished all sample sizes for {experiment_name}.\n")

# Convert the results list to a DataFrame
df_results = pd.DataFrame(results)

# Save the results to a CSV file using the input file name
try:
    df_results.to_csv(output_file_name, index=False)
    print(f"Results saved to {output_file_name}.")
except Exception as e:
    print(f"Error saving results to CSV: {str(e)}")
    print(traceback.format_exc())
