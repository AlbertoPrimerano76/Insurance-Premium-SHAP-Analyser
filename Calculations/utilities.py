import pandas as pd
from datetime import datetime

def _create_mappings_and_transform(df: pd.DataFrame):
    """
    Creates mappings for categorical columns and transforms the DataFrame using those mappings.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing categorical columns to be transformed.
    
    Returns:
    tuple: A tuple containing:
        - df_transformed (pd.DataFrame): The transformed DataFrame with categorical values replaced by integers.
        - mappings (dict): A dictionary containing mappings for each categorical column.
    """
    mappings = {}
    df_transformed = df.copy()

    # Iterate through each column in the DataFrame
    for col in df_transformed.columns:
        if df_transformed[col].dtype == 'object':  # Only process object (string) columns
            unique_values = df_transformed[col].unique()
            col_mapping = {value: idx for idx, value in enumerate(unique_values)}
            mappings[col] = col_mapping
            df_transformed[col] = df_transformed[col].map(col_mapping)
    
    return df_transformed, mappings

def _identify_columns_containing_date(df: pd.DataFrame):
    """
    Identifies all columns in the DataFrame whose names contain the word 'date'.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be analyzed.

    Returns:
    list: A list of column names that contain the word 'date'.
    """
    
    # Identify columns with 'date' in their name (case-insensitive)
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    return date_columns


def map_dataset(df: pd.DataFrame, mappings: dict, target_column):
    """
    Applies the provided mappings to a new DataFrame to transform its categorical values.
    
    Parameters:
    new_df (pd.DataFrame): The new DataFrame to be transformed using the provided mappings.
    mappings (dict): The mappings to apply to the DataFrame. The keys should correspond to column names.
    
    Returns:
    pd.DataFrame: The transformed DataFrame with categorical values replaced by mapped integers.
    """
    
    columns_to_drop = [target_column]

    # Check if 'Name' exists in the DataFrame before attempting to drop it
    if 'Name' in df.columns:
        columns_to_drop.append('Name')
    transformed = df.drop(columns=columns_to_drop)    
    y = df[target_column]  # Although 'y' is not used, it might be useful for further steps
    # Drop any remaining date columns
    date_cols = _identify_columns_containing_date(transformed)
    transformed = transformed.drop(columns=date_cols)
    
     # Identify categorical and numerical columns
    categorical_cols = transformed.select_dtypes(include=['object']).columns
    numerical_cols = transformed.select_dtypes(include=['int64', 'float64']).columns
    
      # Separate categorical and numerical datasets
    transformed_cat = transformed[categorical_cols].copy()
    transformed_num = transformed[numerical_cols].copy()
    
      
    # Iterate through each column in the DataFrame
    for col in transformed_cat.columns:
        if col in mappings:
            try:
                # Apply the mapping; values not found in the mapping will be replaced with NaN
                transformed_cat[col] = transformed_cat[col].map(mappings[col])
            except Exception as e:
                # Log the exception and column name for debugging purposes
                print(f"Error mapping column {col}: {e}")
                transformed_cat[col] = pd.NA
    
    preprocessed = pd.concat([transformed_num, transformed_cat], axis=1)
   
    
    return preprocessed, y

def pre_process_data(sample_data: pd.DataFrame, target_column: str):
    """
    Pre-processes the input data by separating features and target, handling categorical and numerical columns,
    and transforming categorical columns to numeric using mappings.

    Parameters:
    sample_data (pd.DataFrame): The input DataFrame containing the dataset.
    target_column (str): The name of the target column in the dataset.

    Returns:
    tuple: A tuple containing:
        - X_preprocessed (pd.DataFrame): The pre-processed feature dataset with numerical and transformed categorical features.
        - mappings (dict): The mappings used to transform the categorical columns.
    """
    
    # Ensure the target column and necessary columns are present
    if target_column not in sample_data.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the sample data.")
  
    
    # Separate features (X) and target (y)
    # Drop 'Name' if it exists in the DataFrame
    columns_to_drop = [target_column]

    # Check if 'Name' exists in the DataFrame before attempting to drop it
    if 'Name' in sample_data.columns:
        columns_to_drop.append('Name')

    # Drop the columns
    X = sample_data.drop(columns=columns_to_drop)    
    y = sample_data[target_column]  # Although 'y' is not used, it might be useful for further steps

    # Drop any remaining date columns
    date_cols = _identify_columns_containing_date(X)

    X = X.drop(columns=date_cols)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Separate categorical and numerical datasets
    X_cat = X[categorical_cols].copy()
    X_num = X[numerical_cols].copy()

    # Transform categorical columns into numeric using mappings
    df_transformed, mappings = _create_mappings_and_transform(X_cat)
    
    # Combine numerical and transformed categorical columns
    X_preprocessed = pd.concat([X_num, df_transformed], axis=1)
    
    return X_preprocessed,y, mappings
