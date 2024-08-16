import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    """
    Handles missing values in a DataFrame by imputation or dropping.

    Args:
        df: The DataFrame to process.
    """
    df_copy = df.copy()
    
    # Identify columns with missing values
    missing_cols = df_copy.columns[df_copy.isnull().any()].tolist()

    for col in missing_cols:
        if df_copy[col].dtype in ['float64', 'int64']:
            # Impute numerical missing values with the median
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        else:
            # Impute categorical missing values with the most frequent category
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy

def encode_categorical_variables(df,target_column):
    """
    Encodes categorical variables using one-hot encoding or label encoding.

    Args:
        df: The DataFrame to process.
    """
     # Separate features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # One-hot encode categorical features
    features = pd.get_dummies(features, drop_first=True)
    # Combine the preprocessed features with the target column
    preprocessed_df = pd.concat([features, target.reset_index(drop=True)], axis=1)

    return preprocessed_df
    


def scale_numerical_variables(df,target_column):
    """
    Scales numerical variables using standardization or normalization.

    Args:
        df: The DataFrame to process.
    """
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    
    # Identify numerical columns
    numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns

    if not numerical_cols.any():
        print("No numerical columns found to scale.")
        return df
    
    scaler = StandardScaler()  # or MinMaxScaler()
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
    
    preprocessed_df = pd.concat([features, target.reset_index(drop=True)], axis=1)

    
    return preprocessed_df