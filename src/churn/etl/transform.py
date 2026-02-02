"""Transform raw data: inspect schema and fix data types."""

from datetime import timedelta
import re
import pandas as pd
from prefect import flow, task


def to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    # Insert underscore before uppercase letters and lowercase everything
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Handle consecutive uppercase (e.g., "HTTPServer" -> "http_server")
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    # Replace spaces and hyphens with underscores
    s3 = re.sub(r"[\s\-]+", "_", s2)
    # Remove non-alphanumeric characters except underscores
    s4 = re.sub(r"[^\w]", "", s3)
    # Collapse multiple underscores
    s5 = re.sub(r"_+", "_", s4)
    return s5.lower().strip("_")


def standardize_column_names(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Standardize all column names to snake_case.

    Args:
        df: DataFrame with original column names
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with snake_case column names
    """
    df = df.copy()
    rename_map = {col: to_snake_case(col) for col in df.columns}

    # Log renamed columns
    for old, new in rename_map.items():
        if old != new:
            if logger:
                logger.info(f"  {old} -> {new}")

    df.columns = [rename_map[col] for col in df.columns]
    return df

def remove_duplicates(
    df: pd.DataFrame, subset: list[str] | None = None, logger=None
) -> tuple[pd.DataFrame, dict]:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df: DataFrame with potential duplicates
        subset: Columns to consider for identifying duplicates (default: all columns)
        logger: Optional Prefect logger for logging progress

    Returns:
        tuple containing:
            - pd.DataFrame: DataFrame with duplicates removed
            - dict: Info about duplicates removed
    """
    df = df.copy()
    initial_rows = len(df)

    # Find duplicates
    if subset:
        duplicates = df.duplicated(subset=subset, keep="first")
    else:
        duplicates = df.duplicated(keep="first")

    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        df = df[~duplicates]
        if logger:
            logger.info(f"  Removed {n_duplicates} duplicate rows")
    else:
        if logger:
            logger.info("  No duplicate rows found")

    info = {
        "initial_rows": initial_rows,
        "duplicates_removed": int(n_duplicates),
        "final_rows": len(df),
        "subset_columns": subset,
    }

    return df, info


def handle_outliers_and_impossible(
    df: pd.DataFrame, logger=None
) -> tuple[pd.DataFrame, dict]:
    """
    Handle impossible values and outliers in the DataFrame.

    Impossible values:
        - tenure < 0
        - monthly_charges < 0
        - total_charges < 0
        - senior_citizen not in {0, 1}

    Outliers:
        - Uses IQR method (values beyond 1.5 * IQR are capped)

    Args:
        df: DataFrame with potential outliers/impossible values
        logger: Optional Prefect logger for logging progress

    Returns:
        tuple containing:
            - pd.DataFrame: DataFrame with issues handled
            - dict: Info about changes made
    """
    df = df.copy()
    changes = {"impossible_values": {}, "outliers_capped": {}}

    # Define impossible value rules
    impossible_rules = {
        "tenure": {"min": 0, "max": None},
        "monthly_charges": {"min": 0, "max": None},
        "total_charges": {"min": 0, "max": None},
        "senior_citizen": {"allowed": [0, 1]},
    }

    # Fix impossible values
    for col, rules in impossible_rules.items():
        if col not in df.columns:
            continue

        col_changes = {"before": int(df[col].isna().sum())}

        if "allowed" in rules:
            # For discrete allowed values
            invalid_mask = ~df[col].isin(rules["allowed"]) & df[col].notna()
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                df.loc[invalid_mask, col] = pd.NA
                col_changes["invalid_replaced"] = int(n_invalid)
                if logger:
                    logger.info(f"  {col}: {n_invalid} impossible values -> NaN")
        else:
            # For range constraints
            if rules.get("min") is not None:
                invalid_mask = df[col] < rules["min"]
                n_invalid = invalid_mask.sum()
                if n_invalid > 0:
                    df.loc[invalid_mask, col] = pd.NA
                    col_changes["below_min"] = int(n_invalid)
                    if logger:
                        logger.info(f"  {col}: {n_invalid} values below {rules['min']} -> NaN")

            if rules.get("max") is not None:
                invalid_mask = df[col] > rules["max"]
                n_invalid = invalid_mask.sum()
                if n_invalid > 0:
                    df.loc[invalid_mask, col] = pd.NA
                    col_changes["above_max"] = int(n_invalid)
                    if logger:
                        logger.info(f"  {col}: {n_invalid} values above {rules['max']} -> NaN")

        col_changes["after"] = int(df[col].isna().sum())
        if col_changes["after"] > col_changes["before"]:
            changes["impossible_values"][col] = col_changes

    # Handle outliers using IQR for continuous numeric columns
    outlier_columns = ["tenure", "monthly_charges", "total_charges"]

    for col in outlier_columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Cap outliers (winsorization)
        lower_outliers = (df[col] < lower_bound).sum()
        upper_outliers = (df[col] > upper_bound).sum()

        if lower_outliers > 0 or upper_outliers > 0:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            changes["outliers_capped"][col] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "lower_capped": int(lower_outliers),
                "upper_capped": int(upper_outliers),
            }
            if logger:
                logger.info(
                    f"  {col}: capped {lower_outliers} low, {upper_outliers} high outliers"
                )

    if not changes["impossible_values"] and not changes["outliers_capped"]:
        if logger:
            logger.info("  No impossible values or outliers found")

    return df, changes


def inspect_schema(df: pd.DataFrame, logger=None) -> dict:
    """
    Inspect DataFrame schema and identify potential data type issues.

    Args:
        df: Raw DataFrame to inspect
        logger: Optional Prefect logger for logging progress

    Returns:
        dict with schema info and detected issues
    """
    issues = []
    categorical_info = []

    for col in df.columns:
        if df[col].dtype == "object":
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)

            # Check if column contains numeric values stored as text
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            numeric_count = numeric_converted.notna().sum()
            null_after_conversion = (
                numeric_converted.isna().sum() - df[col].isna().sum()
            )

            if numeric_count > len(df) * 0.5:
                issues.append(
                    {
                        "column": col,
                        "issue": "numeric_as_text",
                        "current_dtype": str(df[col].dtype),
                        "numeric_values": int(numeric_count),
                        "failed_conversions": int(null_after_conversion),
                    }
                )
            # Low cardinality = likely categorical
            elif unique_ratio < 0.05 or unique_count <= 10:
                categorical_info.append(
                    {
                        "column": col,
                        "unique_values": int(unique_count),
                        "values": list(df[col].unique()),
                    }
                )

    schema_info = {
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "issues": issues,
        "categorical_candidates": categorical_info,
    }

    if logger:
        logger.info(
            f"Schema inspection: {len(issues)} type issues, {len(categorical_info)} categorical columns"
        )
        for issue in issues:
            logger.info(
                f"  - {issue['column']}: {issue['issue']} ({issue['failed_conversions']} problematic values)"
            )
        for cat in categorical_info:
            logger.info(
                f"  - {cat['column']}: categorical ({cat['unique_values']} unique values)"
            )

    return schema_info


def fix_numeric_types(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Fix numeric data types: convert numeric columns from object to numeric, keep IDs as strings.

    This should run early in the pipeline, before outlier/missing value handling.

    Args:
        df: DataFrame with potential type issues
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with numeric columns converted
    """
    df = df.copy()

    # Columns that should remain as strings (IDs) - snake_case
    id_columns = ["customer_id"]

    # Columns that should be numeric - snake_case
    numeric_columns = ["total_charges", "monthly_charges", "tenure", "senior_citizen"]

    # Fix numeric columns
    for col in numeric_columns:
        if col in df.columns and df[col].dtype == "object":
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum()

            if logger:
                if new_nulls > original_nulls:
                    logger.info(
                        f"  {col}: -> numeric ({new_nulls - original_nulls} values became NaN)"
                    )
                else:
                    logger.info(f"  {col}: -> numeric")

    # Ensure ID columns stay as strings
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            if logger:
                logger.info(f"  {col}: -> string (ID)")

    return df


def convert_to_category_dtype(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Convert categorical columns to pandas category dtype for memory efficiency.

    This should run late in the pipeline, after normalization is complete.

    Args:
        df: DataFrame with normalized categorical columns (still object dtype)
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with categorical columns converted to category dtype
    """
    df = df.copy()

    categorical_columns = [
        "gender",
        "partner",
        "dependents",
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
        "churn",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
            if logger:
                logger.info(f"  {col}: -> category")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_fill: str = "Unknown",
    drop_threshold: float | None = None,
    logger=None,
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df: DataFrame with potential missing values
        numeric_strategy: Strategy for numeric columns - "mean" or "median"
        categorical_fill: Value to fill missing categorical values
        drop_threshold: If set, drop rows with more than this fraction of missing values (0.0-1.0)
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Report initial missing values
    missing_before = df.isna().sum()
    total_missing = missing_before.sum()
    if logger:
        logger.info(f"Total missing values before: {total_missing}")

    if total_missing == 0:
        if logger:
            logger.info("No missing values found.")
        return df

    # Optionally drop rows with too many missing values
    if drop_threshold is not None:
        n_cols = len(df.columns)
        missing_per_row = df.isna().sum(axis=1)
        rows_to_drop = missing_per_row > (drop_threshold * n_cols)
        if rows_to_drop.any():
            n_dropped = rows_to_drop.sum()
            df = df[~rows_to_drop]
            if logger:
                logger.info(f"Dropped {n_dropped} rows with >{drop_threshold:.0%} missing values")

    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if numeric_strategy == "median":
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            if logger:
                logger.info(
                    f"  {col}: filled {missing_count} missing with {numeric_strategy}={fill_value:.2f}"
                )

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            # For category dtype, need to add the fill value to categories first
            if df[col].dtype.name == "category":
                if categorical_fill not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([categorical_fill])
            df[col] = df[col].fillna(categorical_fill)
            if logger:
                logger.info(f"  {col}: filled {missing_count} missing with '{categorical_fill}'")

    # Report final state
    total_remaining = df.isna().sum().sum()
    if logger:
        logger.info(f"Total missing values after: {total_remaining}")

    return df


def normalize_categorical_columns(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Normalize categorical columns: trim whitespace, unify labels, standardize casing.

    Transformations applied:
        - Strip leading/trailing whitespace
        - Unify "No internet service" / "No phone service" → "No"
        - Apply title case for consistency (e.g., "female" → "Female")

    Args:
        df: DataFrame with categorical columns
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with normalized categorical columns
    """
    df = df.copy()

    # Values to unify to "No"
    no_service_values = ["No internet service", "No phone service"]

    # Get all object (string) columns
    string_cols = df.select_dtypes(include=["object"]).columns

    for col in string_cols:
        original_values = df[col].unique()

        # 1. Trim whitespace
        df[col] = df[col].str.strip()

        # 2. Unify "No internet service" / "No phone service" → "No"
        df[col] = df[col].replace(no_service_values, "No")

        # 3. Apply title case for consistency
        df[col] = df[col].str.title()

        # Report changes
        new_values = df[col].unique()
        changed = set(original_values) != set(new_values)
        if changed:
            if logger:
                logger.info(f"  {col}: normalized values")

    return df


def create_binary_flags(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    logger=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Create binary flag columns (1/0) from categorical columns with exactly 2 unique values.

    Automatically detects columns with 2 unique values and creates _flag columns.
    Known mappings (Yes/No, Male/Female) use predefined order, others are sorted alphabetically.

    Args:
        df: DataFrame with categorical columns
        exclude_columns: Columns to skip (e.g., IDs)
        logger: Optional Prefect logger for logging progress

    Returns:
        tuple containing:
            - pd.DataFrame: DataFrame with new flag columns
            - dict: Mapping info for each column
    """
    df = df.copy()
    mappings = {}

    if exclude_columns is None:
        exclude_columns = ["customer_id"]

    # Known binary mappings (value that maps to 1 first, value that maps to 0 second)
    known_mappings = {
        ("Yes", "No"): {"Yes": 1, "No": 0},
        ("No", "Yes"): {"Yes": 1, "No": 0},
        ("Male", "Female"): {"Male": 1, "Female": 0},
        ("Female", "Male"): {"Male": 1, "Female": 0},
    }

    # Find categorical/object columns with exactly 2 unique values
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col in exclude_columns:
            continue

        # Get column data as string
        col_data = df[col].astype(str) if df[col].dtype.name == "category" else df[col]

        # Get unique values (excluding NaN)
        unique_values = tuple(sorted(col_data.dropna().unique()))

        # Skip if not exactly 2 unique values
        if len(unique_values) != 2:
            continue

        # Check if it's a known mapping
        if unique_values in known_mappings:
            mapping = known_mappings[unique_values]
        else:
            # Create alphabetical mapping: first=0, second=1
            mapping = {unique_values[0]: 0, unique_values[1]: 1}

        # Create flag column
        flag_col = f"{col}_flag"
        df[flag_col] = col_data.map(mapping).astype("Int64")
        mappings[flag_col] = {"source_column": col, "mapping": mapping}

        mapping_str = ", ".join(f"{k}={v}" for k, v in mapping.items())
        if logger:
            logger.info(f"  {col} -> {flag_col}: {mapping_str}")

    if not mappings:
        if logger:
            logger.info("  No columns with exactly 2 unique values found")

    return df, mappings


def calculate_total_services(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Calculate total number of services per customer.

    Creates a 'total_services' column by counting how many services each customer has.
    A service is considered active if its value is not 'No'.

    Service columns checked:
        - phone_service, multiple_lines, internet_service
        - online_security, online_backup, device_protection
        - tech_support, streaming_tv, streaming_movies

    Args:
        df: DataFrame with service columns
        logger: Optional Prefect logger for logging progress

    Returns:
        DataFrame with 'total_services' column added
    """
    df = df.copy()

    service_columns = [
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    # Filter to only columns that exist in the DataFrame
    existing_service_cols = [col for col in service_columns if col in df.columns]

    if not existing_service_cols:
        if logger:
            logger.info("  No service columns found")
        return df

    # Count services: 1 if value is not 'No', 0 otherwise
    def has_service(val) -> int:
        if pd.isna(val):
            return 0
        return 1 if str(val).lower() != "no" else 0

    # Calculate total services - convert to string first to handle categorical dtype
    df["total_services"] = (
        pd.concat(
            [df[col].astype(str).apply(has_service) for col in existing_service_cols],
            axis=1,
        )
        .sum(axis=1)
        .astype(int)
    )

    if logger:
        logger.info(
            f"  Counted services from {len(existing_service_cols)} columns: {existing_service_cols}"
        )
        logger.info(
            f"  total_services range: {df['total_services'].min()} - {df['total_services'].max()}"
        )

    return df


def encode_multi_class_categories(
    df: pd.DataFrame,
    min_unique: int = 3,
    max_unique: int = 10,
    exclude_columns: list[str] | None = None,
    logger=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Create encoded columns for categorical columns with 3-10 unique values.

    Automatically detects categorical/object columns with unique value count
    between min_unique and max_unique, then creates new _encoded columns.

    Args:
        df: DataFrame with categorical columns
        min_unique: Minimum unique values to encode (default: 3)
        max_unique: Maximum unique values to encode (default: 10)
        exclude_columns: Columns to skip (e.g., IDs)
        logger: Optional Prefect logger for logging progress

    Returns:
        tuple containing:
            - pd.DataFrame: DataFrame with new encoded columns
            - dict: Mapping info for each column {column: {value: code}}
    """
    df = df.copy()
    mappings = {}

    if exclude_columns is None:
        exclude_columns = ["customer_id"]

    # Find categorical/object columns with 3-10 unique values
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col in exclude_columns:
            continue

        # Get column data as string
        col_data = df[col].astype(str) if df[col].dtype.name == "category" else df[col]

        # Get unique values (excluding NaN)
        unique_values = sorted(col_data.dropna().unique())
        n_unique = len(unique_values)

        # Skip if outside range
        if n_unique < min_unique or n_unique > max_unique:
            continue

        # Create mapping: value -> 1, 2, 3, ...
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

        # Create new encoded column
        encoded_col = f"{col}_encoded"
        df[encoded_col] = col_data.map(mapping).astype("Int64")
        mappings[encoded_col] = {"source_column": col, "mapping": mapping}

        mapping_str = ", ".join(f"{v}={k}" for v, k in mapping.items())
        if logger:
            logger.info(f"  {col} -> {encoded_col}: {mapping_str}")

    if not mappings:
        if logger:
            logger.info("  No columns with 3-10 unique values found")

    return df, mappings


def validate_transformed_data(df: pd.DataFrame, logger=None) -> dict:
    """
    Validate the transformed DataFrame meets all expectations.

    Checks:
        - No missing values in critical columns
        - Data types are correct
        - Numeric columns within expected ranges
        - Categorical columns have valid values
        - Binary flags are 0/1 only
        - No duplicate rows

    Args:
        df: Transformed DataFrame to validate
        logger: Optional Prefect logger for logging progress

    Returns:
        dict with validation results and any issues found
    """
    issues = []
    warnings = []

    # === 1. Check for missing values ===
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        for col, count in cols_with_missing.items():
            issues.append(f"Column '{col}' has {count} missing values")

    # === 2. Check data types ===
    expected_types = {
        # Numeric columns
        "tenure": ["int64", "float64", "Int64", "Float64"],
        "monthly_charges": ["float64", "Float64"],
        "total_charges": ["float64", "Float64"],
        "senior_citizen": ["int64", "float64", "Int64", "Float64"],
        "total_services": ["int64", "Int64"],
        # ID column
        "customer_id": ["object", "string"],
        # Categorical columns
        "gender": ["category"],
        "partner": ["category"],
        "dependents": ["category"],
        "phone_service": ["category"],
        "internet_service": ["category"],
        "contract": ["category"],
        "churn": ["category"],
    }

    for col, valid_types in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type not in valid_types:
                warnings.append(
                    f"Column '{col}' has type '{actual_type}', expected one of {valid_types}"
                )

    # === 3. Check numeric ranges ===
    numeric_ranges = {
        "tenure": {"min": 0, "max": 100},  # months
        "monthly_charges": {"min": 0, "max": 500},
        "total_charges": {"min": 0, "max": 10000},
        "senior_citizen": {"min": 0, "max": 1},
        "total_services": {"min": 0, "max": 9},
    }

    for col, ranges in numeric_ranges.items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < ranges["min"]:
                issues.append(
                    f"Column '{col}' has values below {ranges['min']} (min: {col_min})"
                )
            if col_max > ranges["max"]:
                warnings.append(
                    f"Column '{col}' has values above {ranges['max']} (max: {col_max})"
                )

    # === 4. Check categorical values ===
    expected_categories = {
        "gender": ["Male", "Female"],
        "partner": ["Yes", "No"],
        "dependents": ["Yes", "No"],
        "phone_service": ["Yes", "No"],
        "churn": ["Yes", "No"],
        "internet_service": ["Dsl", "Fiber Optic", "No"],
        "contract": ["Month-To-Month", "One Year", "Two Year"],
    }

    for col, valid_values in expected_categories.items():
        if col in df.columns:
            # Get actual unique values
            if df[col].dtype.name == "category":
                actual_values = set(df[col].cat.categories)
            else:
                actual_values = set(df[col].dropna().unique())

            # Normalize for comparison (title case)
            valid_set = {v.title() if isinstance(v, str) else v for v in valid_values}
            actual_set = {str(v).title() if pd.notna(v) else v for v in actual_values}

            unexpected = actual_set - valid_set
            if unexpected:
                warnings.append(f"Column '{col}' has unexpected values: {unexpected}")

    # === 5. Check total_services is valid ===
    if "total_services" in df.columns:
        ts_min = df["total_services"].min()
        ts_max = df["total_services"].max()
        if ts_min < 0:
            issues.append(f"total_services has negative values (min: {ts_min})")
        if ts_max > 9:
            warnings.append(f"total_services exceeds expected max of 9 (max: {ts_max})")

    # === 6. Check for duplicates ===
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings.append(f"Found {n_duplicates} duplicate rows")

    # === 7. Check customer_id uniqueness ===
    if "customer_id" in df.columns:
        n_unique_ids = df["customer_id"].nunique()
        if n_unique_ids != len(df):
            issues.append(
                f"customer_id is not unique: {n_unique_ids} unique vs {len(df)} rows"
            )

    # === Build validation result ===
    is_valid = len(issues) == 0
    validation_result = {
        "is_valid": is_valid,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_value_columns": len(cols_with_missing),
            "duplicate_rows": int(n_duplicates),
        },
    }

    # Log results
    if logger:
        if is_valid:
            logger.info("  All validations passed")
        else:
            logger.info(f"  {len(issues)} issue(s) found:")
            for issue in issues:
                logger.info(f"    - {issue}")

        if warnings:
            logger.info(f"  {len(warnings)} warning(s):")
            for warning in warnings:
                logger.info(f"    - {warning}")

        logger.info(f"  Summary: {len(df)} rows, {len(df.columns)} columns")

    return validation_result


@task(
    name="transform-data",
    retry_delay_seconds=[2, 4, 8],
    cache_key_fn=lambda params: params["raw_hash"],
    cache_expiration=timedelta(days=3),
)
def transform_data(
    df: pd.DataFrame, raw_hash: str, logger=None
) -> tuple[pd.DataFrame, dict]:
    """
    Main transform function: clean and normalize data.

    Produces a clean canonical table with normalized types and categories.
    Model-specific encoding (one-hot, label encoding) should happen in
    training pipelines, not here.

    Steps:
        1. Standardize column names to snake_case
        2. Fix numeric types (convert numeric columns)
        3. Remove duplicate rows
        4. Normalize categorical values
        5. Handle missing values
        6. Handle outliers and impossible values
        7. Convert to category dtype
        8. Feature engineering (total_services only)
        9. Validate transformed data

    Args:
        df: Raw DataFrame from extract step
        logger: Optional Prefect logger for logging progress

    Returns:
        tuple containing:
            - pd.DataFrame: Cleaned DataFrame with correct data types
            - dict: Comprehensive transformation log

    Outputs saved to output_dir:
        - churn_cleaned.csv: Transformed DataFrame
        - transformation_log.json: Comprehensive transformation log
    """
    transformation_log = {
        "initial_shape": {"rows": len(df), "columns": len(df.columns)},
        "steps": {},
    }

    # 1. Standardize column names
    if logger:
        logger.info("=== 1. Standardizing Column Names ===")
    df = standardize_column_names(df, logger=logger)

    # 2. Fix numeric types (early, before other processing)
    if logger:
        logger.info("=== 2. Fixing Numeric Types ===")
    df = fix_numeric_types(df, logger=logger)

    # 3. Remove duplicates
    if logger:
        logger.info("=== 3. Removing Duplicates ===")
    df, duplicates_info = remove_duplicates(df, logger=logger)
    transformation_log["steps"]["remove_duplicates"] = duplicates_info

    # 4. Normalize categorical values
    if logger:
        logger.info("=== 4. Normalizing Categorical Values ===")
    df = normalize_categorical_columns(df, logger=logger)
    transformation_log["steps"]["normalize_categorical"] = {
        "transformations": [
            "Trimmed whitespace",
            "Unified 'No internet service'/'No phone service' -> 'No'",
            "Applied title case",
        ]
    }

    # 5. Handle missing values
    if logger:
        logger.info("=== 5. Handling Missing Values ===")
    df = handle_missing_values(df, logger=logger)

    # 6. Handle outliers and impossible values
    if logger:
        logger.info("=== 6. Handling Outliers & Impossible Values ===")
    df, outliers_info = handle_outliers_and_impossible(df, logger=logger)
    transformation_log["steps"]["outliers_impossible"] = outliers_info

    # 7. Convert to category dtype (memory efficiency)
    if logger:
        logger.info("=== 7. Converting to Category Dtype ===")
    df = convert_to_category_dtype(df, logger=logger)

    # 8. Feature engineering (total_services only - no pre-encoding)
    if logger:
        logger.info("=== 8. Feature Engineering (Total Services) ===")
    df = calculate_total_services(df, logger=logger)
    transformation_log["steps"]["feature_engineering"] = {
        "total_services": "Count of active services per customer"
    }

    # 9. Validate transformed data
    if logger:
        logger.info("=== 9. Validating Transformed Data ===")
    validation_result = validate_transformed_data(df, logger=logger)
    transformation_log["steps"]["validation"] = validation_result

    transformation_log["final_shape"] = {"rows": len(df), "columns": len(df.columns)}
    transformation_log["final_columns"] = list(df.columns)

    if logger:
        logger.info("=== Final Schema ===")
        logger.info(f"\n{df.dtypes}")
        logger.info(
            f"Transformation complete. Final shape: {len(df)} rows, {len(df.columns)} columns."
        )

    return df, transformation_log
