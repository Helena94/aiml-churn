"""Shared pytest fixtures for ETL tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    """Sample raw DataFrame mimicking Kaggle telco churn data."""
    return pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
            "gender": ["Female", "Male", "Male"],
            "SeniorCitizen": [0, 0, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "No"],
            "tenure": [1, 34, 2],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No"],
            "InternetService": ["DSL", "DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes", "Yes"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["No", "No", "No"],
            "StreamingTV": ["No", "No", "No"],
            "StreamingMovies": ["No", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", "1889.5", "108.15"],
            "Churn": ["No", "No", "Yes"],
        }
    )


@pytest.fixture
def sample_cleaned_df():
    """Sample cleaned DataFrame after transformation."""
    return pd.DataFrame(
        {
            "customer_id": ["7590-VHVEG", "5575-GNVDE"],
            "gender": pd.Categorical(["Female", "Male"]),
            "senior_citizen": [0, 0],
            "partner": pd.Categorical(["Yes", "No"]),
            "dependents": pd.Categorical(["No", "No"]),
            "tenure": [1, 34],
            "phone_service": pd.Categorical(["No", "Yes"]),
            "internet_service": pd.Categorical(["Dsl", "Dsl"]),
            "contract": pd.Categorical(["Month-To-Month", "One Year"]),
            "monthly_charges": [29.85, 56.95],
            "total_charges": [29.85, 1889.5],
            "churn": pd.Categorical(["No", "No"]),
            "total_services": [2, 4],
        }
    )
