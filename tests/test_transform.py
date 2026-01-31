"""Unit tests for ETL transform functions."""

import pandas as pd
import pytest

from churn.etl.transform import (
    calculate_total_services,
    convert_to_category_dtype,
    create_binary_flags,
    encode_multi_class_categories,
    fix_numeric_types,
    handle_missing_values,
    handle_outliers_and_impossible,
    inspect_schema,
    normalize_categorical_columns,
    remove_duplicates,
    standardize_column_names,
    to_snake_case,
    validate_transformed_data,
)


class TestToSnakeCase:
    """Tests for to_snake_case function."""

    def test_camel_case(self):
        assert to_snake_case("CustomerID") == "customer_id"
        assert to_snake_case("firstName") == "first_name"

    def test_pascal_case(self):
        assert to_snake_case("TotalCharges") == "total_charges"
        assert to_snake_case("MonthlyCharges") == "monthly_charges"

    def test_with_spaces(self):
        assert to_snake_case("Total Charges") == "total_charges"
        assert to_snake_case("Customer ID") == "customer_id"

    def test_with_hyphens(self):
        assert to_snake_case("total-charges") == "total_charges"
        assert to_snake_case("customer-id") == "customer_id"

    def test_already_snake_case(self):
        assert to_snake_case("customer_id") == "customer_id"
        assert to_snake_case("total_charges") == "total_charges"

    def test_consecutive_uppercase(self):
        assert to_snake_case("HTTPServer") == "http_server"
        assert to_snake_case("XMLParser") == "xml_parser"

    def test_mixed_formats(self):
        assert to_snake_case("myHTTPServer") == "my_http_server"


class TestStandardizeColumnNames:
    """Tests for standardize_column_names function."""

    def test_standardizes_columns(self):
        df = pd.DataFrame({"CustomerID": [1], "TotalCharges": [100.0]})
        result = standardize_column_names(df)
        assert list(result.columns) == ["customer_id", "total_charges"]

    def test_preserves_data(self):
        df = pd.DataFrame({"CustomerID": [1, 2], "TotalCharges": [100.0, 200.0]})
        result = standardize_column_names(df)
        assert result["customer_id"].tolist() == [1, 2]
        assert result["total_charges"].tolist() == [100.0, 200.0]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"CustomerID": [1]})
        standardize_column_names(df)
        assert "CustomerID" in df.columns


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""

    def test_removes_full_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        result, info = remove_duplicates(df)
        assert len(result) == 2
        assert info["duplicates_removed"] == 1

    def test_removes_subset_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        result, info = remove_duplicates(df, subset=["a"])
        assert len(result) == 2
        assert info["duplicates_removed"] == 1

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result, info = remove_duplicates(df)
        assert len(result) == 3
        assert info["duplicates_removed"] == 0

    def test_keeps_first_duplicate(self):
        df = pd.DataFrame({"a": [1, 1], "b": [10, 10]})  # Full duplicate rows
        result, info = remove_duplicates(df)
        assert len(result) == 1
        assert result["b"].tolist() == [10]


class TestHandleOutliersAndImpossible:
    """Tests for handle_outliers_and_impossible function."""

    def test_negative_tenure_becomes_nan(self):
        df = pd.DataFrame({"tenure": [-5, 10, 20]})
        result, changes = handle_outliers_and_impossible(df)
        assert pd.isna(result["tenure"].iloc[0])
        assert result["tenure"].iloc[1] == 10

    def test_invalid_senior_citizen_becomes_nan(self):
        df = pd.DataFrame({"senior_citizen": [0, 1, 5]})
        result, changes = handle_outliers_and_impossible(df)
        assert pd.isna(result["senior_citizen"].iloc[2])
        assert result["senior_citizen"].iloc[0] == 0
        assert result["senior_citizen"].iloc[1] == 1

    def test_outliers_capped(self):
        df = pd.DataFrame({"tenure": [0, 10, 20, 30, 40, 50, 1000]})
        result, changes = handle_outliers_and_impossible(df)
        # IQR capping should reduce the outlier
        assert result["tenure"].max() < 1000


class TestInspectSchema:
    """Tests for inspect_schema function."""

    def test_detects_numeric_as_text(self):
        df = pd.DataFrame({"amount": ["100", "200", "300"]})
        schema = inspect_schema(df)
        issues = schema["issues"]
        assert len(issues) == 1
        assert issues[0]["issue"] == "numeric_as_text"

    def test_detects_categorical_columns(self):
        df = pd.DataFrame({"status": ["Yes", "No", "Yes", "No"] * 25})
        schema = inspect_schema(df)
        assert len(schema["categorical_candidates"]) == 1
        assert schema["categorical_candidates"][0]["column"] == "status"

    def test_returns_shape(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        schema = inspect_schema(df)
        assert schema["shape"]["rows"] == 2
        assert schema["shape"]["columns"] == 2


class TestFixNumericTypes:
    """Tests for fix_numeric_types function."""

    def test_converts_total_charges(self):
        df = pd.DataFrame({"total_charges": ["100.5", "200.0", " "]})
        result = fix_numeric_types(df)
        assert result["total_charges"].dtype in ["float64", "Float64"]
        assert result["total_charges"].iloc[0] == 100.5

    def test_keeps_customer_id_as_string(self):
        df = pd.DataFrame({"customer_id": [123, 456]})
        result = fix_numeric_types(df)
        assert result["customer_id"].dtype == "object"

    def test_handles_non_numeric_values(self):
        df = pd.DataFrame({"total_charges": ["100", "invalid", "300"]})
        result = fix_numeric_types(df)
        assert pd.isna(result["total_charges"].iloc[1])


class TestConvertToCategoryDtype:
    """Tests for convert_to_category_dtype function."""

    def test_converts_categorical_columns(self):
        df = pd.DataFrame({"gender": ["Male", "Female"], "churn": ["Yes", "No"]})
        result = convert_to_category_dtype(df)
        assert result["gender"].dtype.name == "category"
        assert result["churn"].dtype.name == "category"

    def test_ignores_non_categorical_columns(self):
        df = pd.DataFrame({"tenure": [10, 20], "gender": ["Male", "Female"]})
        result = convert_to_category_dtype(df)
        assert result["tenure"].dtype != "category"


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_fills_numeric_with_median(self):
        df = pd.DataFrame({"tenure": [10.0, None, 30.0]})
        result = handle_missing_values(df, numeric_strategy="median")
        assert result["tenure"].iloc[1] == 20.0  # median of 10 and 30

    def test_fills_numeric_with_mean(self):
        df = pd.DataFrame({"tenure": [10.0, None, 20.0]})
        result = handle_missing_values(df, numeric_strategy="mean")
        assert result["tenure"].iloc[1] == 15.0  # mean of 10 and 20

    def test_fills_categorical_with_unknown(self):
        df = pd.DataFrame({"status": ["Yes", None, "No"]})
        result = handle_missing_values(df, categorical_fill="Unknown")
        assert result["status"].iloc[1] == "Unknown"

    def test_no_missing_values_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = handle_missing_values(df)
        assert result["a"].tolist() == [1, 2, 3]


class TestNormalizeCategoricalColumns:
    """Tests for normalize_categorical_columns function."""

    def test_trims_whitespace(self):
        df = pd.DataFrame({"status": ["  Yes  ", "No  "]})
        result = normalize_categorical_columns(df)
        assert result["status"].tolist() == ["Yes", "No"]

    def test_unifies_no_service_values(self):
        df = pd.DataFrame({"internet": ["No internet service", "Yes"]})
        result = normalize_categorical_columns(df)
        assert result["internet"].iloc[0] == "No"

    def test_applies_title_case(self):
        df = pd.DataFrame({"status": ["yes", "NO", "maybe"]})
        result = normalize_categorical_columns(df)
        assert result["status"].tolist() == ["Yes", "No", "Maybe"]


class TestCreateBinaryFlags:
    """Tests for create_binary_flags function."""

    def test_creates_yes_no_flag(self):
        df = pd.DataFrame({"churn": ["Yes", "No", "Yes"]})
        result, mappings = create_binary_flags(df)
        assert "churn_flag" in result.columns
        assert result["churn_flag"].tolist() == [1, 0, 1]

    def test_creates_gender_flag(self):
        df = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
        result, mappings = create_binary_flags(df)
        assert "gender_flag" in result.columns
        assert result["gender_flag"].tolist() == [1, 0, 1]

    def test_excludes_customer_id(self):
        df = pd.DataFrame({"customer_id": ["A", "B"]})
        result, mappings = create_binary_flags(df)
        assert "customer_id_flag" not in result.columns

    def test_skips_multi_value_columns(self):
        df = pd.DataFrame({"contract": ["Month", "Year", "Two Year"]})
        result, mappings = create_binary_flags(df)
        assert "contract_flag" not in result.columns


class TestCalculateTotalServices:
    """Tests for calculate_total_services function."""

    def test_counts_active_services(self):
        df = pd.DataFrame(
            {
                "phone_service": ["Yes", "No"],
                "internet_service": ["Fiber Optic", "No"],
                "online_security": ["Yes", "No"],
            }
        )
        result = calculate_total_services(df)
        assert result["total_services"].iloc[0] == 3
        assert result["total_services"].iloc[1] == 0

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"phone_service": ["Yes", "No"]})
        result = calculate_total_services(df)
        assert "total_services" in result.columns

    def test_no_service_columns(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = calculate_total_services(df)
        assert "total_services" not in result.columns


class TestEncodeMultiClassCategories:
    """Tests for encode_multi_class_categories function."""

    def test_encodes_contract_column(self):
        df = pd.DataFrame({"contract": ["Month", "One Year", "Two Year"]})
        result, mappings = encode_multi_class_categories(df)
        assert "contract_encoded" in result.columns
        assert "contract_encoded" in mappings

    def test_skips_binary_columns(self):
        df = pd.DataFrame({"churn": ["Yes", "No"]})
        result, mappings = encode_multi_class_categories(df)
        assert "churn_encoded" not in result.columns

    def test_excludes_customer_id(self):
        df = pd.DataFrame({"customer_id": ["A", "B", "C", "D"]})
        result, mappings = encode_multi_class_categories(df)
        assert "customer_id_encoded" not in result.columns


class TestValidateTransformedData:
    """Tests for validate_transformed_data function."""

    def test_valid_data_passes(self):
        df = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "tenure": [10, 20],
                "monthly_charges": [50.0, 60.0],
                "total_charges": [500.0, 1200.0],
                "senior_citizen": [0, 1],
                "total_services": [3, 5],
            }
        )
        result = validate_transformed_data(df)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_detects_missing_values(self):
        df = pd.DataFrame({"tenure": [10, None, 20]})
        result = validate_transformed_data(df)
        assert any("missing" in issue.lower() for issue in result["issues"])

    def test_detects_duplicate_customer_ids(self):
        df = pd.DataFrame({"customer_id": ["A", "A", "B"]})
        result = validate_transformed_data(df)
        assert any("unique" in issue.lower() for issue in result["issues"])

    def test_returns_summary_stats(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_transformed_data(df)
        assert "summary" in result
        assert result["summary"]["total_rows"] == 2
