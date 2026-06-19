"""
Data preprocessing pipeline for the UCI Student Performance dataset (Math course).
Prepares features for a neural network regression model predicting final grade (G3).

Student Dataset
       ↓
Data Preprocessing
       ↓
MLP Regressor
       ↓
Predicted Grade
       ↓
Fuzzy Logic System
       ↓
Risk Classification
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(filepath: str = "student-mat.csv"):
    """
    Load the dataset, encode categoricals, scale features, and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load dataset (UCI format uses semicolon as delimiter)
    df = pd.read_csv(filepath, sep=";")

    # Drop any rows with missing values
    df = df.dropna()

    # Encode all categorical (non-numeric) columns with LabelEncoder
    # (e.g. school, sex, address, Mjob, Fjob, schoolsup, etc.)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Separate target variable (G3) from input features
    y = df["G3"]
    X = df.drop(columns=["G3"])

    # Normalize features to [0, 1] range for neural network input
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # 85/15 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("Data preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")
