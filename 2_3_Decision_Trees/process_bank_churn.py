import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, Any

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary columns from the DataFrame.
    """
    columns_to_remove = ['id', 'CustomerId', 'Surname']
    return df.drop(columns=columns_to_remove, inplace=False)

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)

def separate_features_and_target(df: pd.DataFrame, target_column: str = 'Exited') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates input features and target variable from the DataFrame.
    """
    inputs = df.drop(columns=[target_column])
    target = df[target_column]
    return inputs, target

def identify_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identifies numeric and categorical columns in the DataFrame.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: list, apply_scaling: bool) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numeric features using MinMaxScaler if apply_scaling is True.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    if apply_scaling:
        train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
        val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler

def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list]:
    """
    One-hot encodes categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    return train_df, val_df, encoder, encoded_cols

def preprocess_new_data(raw_test_df: pd.DataFrame, scaler, encoder, scale_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Prepares the new test dataset by applying preprocessing steps such as column dropping,
    feature scaling, and encoding.

    Parameters:
    raw_test_df (pd.DataFrame): The raw test dataset.
    scale_numeric (bool): Whether to apply scaling to numeric features.
    scaler (MinMaxScaler): The fitted scaler for numeric features.
    encoder (OneHotEncoder): The fitted encoder for categorical features.
    
    Returns:
    Dict[str, pd.DataFrame]: A dictionary containing the processed test features under the key 'X_test'.
    
    """
    df = drop_unused_columns(raw_test_df)
    numeric_cols, categorical_cols = identify_column_types(df)
    
    if scale_numeric:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    
    input_cols = numeric_cols + encoded_cols
    return {
        'X_test': df[input_cols]
    }
    
def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, Any]:
    """
    Prepares the dataset by applying preprocessing steps such as column dropping,
    splitting by training/validation, feature scaling, and encoding.
    """
    df = drop_unused_columns(raw_df)
    train_df, val_df = split_data(df)
    train_inputs, train_targets = separate_features_and_target(train_df)
    val_inputs, val_targets = separate_features_and_target(val_df)
    numeric_cols, categorical_cols = identify_column_types(train_inputs)
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols, scale_numeric)
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)
    input_cols = numeric_cols + encoded_cols
    return {
        'X_train': train_inputs[input_cols],
        'y_train': train_targets,
        'X_val': val_inputs[input_cols],
        'y_val': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }
