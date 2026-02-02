ID_COLUMNS = ["customer_id"]

CATEGORICAL_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

NUMERICAL_COLUMNS = [
    "monthly_charges",
    "total_charges",
    "total_services"
]

TARGET_COLUMN = ["churn"]