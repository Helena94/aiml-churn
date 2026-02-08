import pandas as pd

from churn.features.builder import split_train_test


def run_experiment():
    model_name = "linear_regression"
    data = pd.read_csv("../data/churn_cleaned.csv")

    train_data, test_data = split_train_test(data)
