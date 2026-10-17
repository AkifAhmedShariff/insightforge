
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


def get_outliers(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Return rows whose value falls outside the IQR fences."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * q1

    return df[(df[column] < lower) | (df[column] > upper)]


def get_category_totals(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    top_n: int = 5,
) -> list:
    """Return the top categories by total value, largest first, with share of total."""
    _validate_numeric_column(df, value_column)

    totals = df.groupby(category_column)[value_column].sum().sort_values()
    totals = totals.head(top_n)
    grand_total = totals.sum()

    return [
        {
            "category": category,
            "total": float(total),
            "share": float(total / grand_total * 100),
        }
        for category, total in totals.items()
    ]


def get_missing_value_percentages(df: pd.DataFrame) -> dict:
    """Return the percentage of missing values in each column."""
    if df.empty:
        return {column: 0.0 for column in df.columns}

    percentages = df.isnull().sum() / len(df) * 100
    return {column: round(float(value), 2) for column, value in percentages.items()}
