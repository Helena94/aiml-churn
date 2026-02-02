from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features.schema import CATEGORICAL_COLUMNS, NUMERIC_COLS, CATEGORICAL_COLS, NUMERICAL_COLUMNS


def build_preprocessor():
    """
    Returns a sklearn ColumnTransformer
    (NOT fitted).
    """

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop"
    )

    return preprocessor
