import pandas as pd
from sklearn.model_selection import train_test_split

from churn.features.preprocessing import build_preprocessor
from churn.features.schema import ID_COLUMNS, TARGET_COLUMN


def split_train_test(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Splits DataFrame into training and test sets, stratified by target.
    Returns X_train, X_test, y_train, y_test.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def split_features_target(df: pd.DataFrame):
    """
    Returns X, y (raw, unencoded).
    """
    X = df.drop(columns=TARGET_COLUMN + ID_COLUMNS)
    y = df[TARGET_COLUMN[0]]

    return X, y


def preprocess_features(X_train, X_test):
    """
    Fits the preprocessor on X_train, transforms both X_train and X_test.
    Returns numeric arrays ready for model training.
    """
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    return X_train_enc, X_test_enc


def build_features(df: pd.DataFrame):
    """
    Used for inference (no y).
    """
    X = df.drop(columns=ID_COLUMNS + TARGET_COLUMN, errors="ignore")
    return X
