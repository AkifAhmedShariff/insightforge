
import pandas as pd


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """Return basic data-quality metrics for a DataFrame."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_value_count": int(df.isnull().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Validate a column, allowing an all-missing column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    if df[column].isna().all():
        return

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column is not numeric: {column}")


def get_numeric_statistics(
    df: pd.DataFrame,
    column: str,
) -> dict:
    """Return numeric statistics for a column, excluding missing values."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def get_highest_value_row(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return the row containing the highest non-missing numeric value."""
    _validate_numeric_column(df, column)

    if df[column].dropna().empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return df.loc[df[column].idxmax()]
