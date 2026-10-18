
import pandas as pd


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """Return basic data-quality metrics for a DataFrame."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_value_count": int(df.isnull().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def get_moving_average(df: pd.DataFrame, column: str, window: int = 3) -> list:
    """Return the rolling mean of a numeric column over the given window size."""
    _validate_numeric_column(df, column)

    if window < 1:
        raise ValueError("Window must be at least 1")

    averages = df[column].rolling(window=window).mean()
    return [None if pd.isna(value) else round(float(value), 2) for value in averages]


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


def get_median_by_category(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> dict:
    """Return the median value for each category."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    medians = df.groupby(category_column)[value_column].median()
    return {category: round(float(value), 2) for category, value in medians.items()}


def get_value_range(df: pd.DataFrame, column: str) -> float:
    """Return the spread of a numeric column: max minus min."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return float(values.max() - values.min())


def get_correlation(df: pd.DataFrame, column_a: str, column_b: str) -> float:
    """Return the Pearson correlation between two numeric columns."""
    _validate_numeric_column(df, column_a)
    _validate_numeric_column(df, column_b)

    pairs = df[[column_a, column_b]].dropna()

    if len(pairs) < 2:
        raise ValueError("Need at least two complete rows to compute correlation")

    return round(float(pairs[column_a].corr(pairs[column_b], method="pearson")), 4)


def get_highest_value_row(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return the row containing the highest non-missing numeric value."""
    _validate_numeric_column(df, column)

    if df[column].dropna().empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return df.loc[df[column].idxmax()]


def get_percentile(df: pd.DataFrame, column: str, percentile: float) -> float:
    """Return the given percentile (0 to 100, inclusive) of a numeric column."""
    _validate_numeric_column(df, column)

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return float(values.quantile(percentile / 100))


def get_percentage_of_total(df: pd.DataFrame, column: str) -> list:
    """Return each value as a percentage of the column total."""
    _validate_numeric_column(df, column)

    total = df[column].sum()

    if not total:
        return [None if pd.isna(value) else 0.0 for value in df[column]]

    shares = df[column] / total * 100
    return [None if pd.isna(value) else round(float(value), 2) for value in shares]


def drop_duplicate_rows(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Return a copy without duplicate rows, keeping the first occurrence of each."""
    if subset:
        missing = [column for column in subset if column not in df.columns]
        if missing:
            raise KeyError(f"Column not found: {missing[0]}")

    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


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


def get_z_scores(df: pd.DataFrame, column: str) -> list:
    """Return the z-score of each value: (value - mean) / standard deviation."""
    _validate_numeric_column(df, column)

    mean = df[column].mean()
    scale = df[column].std()

    if not scale:
        return [None if pd.isna(value) else 0.0 for value in df[column]]

    scores = (df[column] - mean) / scale
    return [None if pd.isna(value) else round(float(value), 4) for value in scores]


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


def get_monthly_totals(df: pd.DataFrame, date_column: str, value_column: str) -> dict:
    """Return the total value per month, keyed by "YYYY-MM"."""
    _validate_numeric_column(df, value_column)

    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    months = dates.dt.strftime("%Y-%m")
    totals = df[value_column].groupby(months).sum()

    return {month: float(total) for month, total in totals.items()}


def count_values_above(df: pd.DataFrame, column: str, threshold: float) -> int:
    """Return how many values are strictly greater than the threshold."""
    _validate_numeric_column(df, column)

    return int((df[column] > threshold).sum())


def get_top_n_rows(df: pd.DataFrame, column: str, n: int = 5) -> pd.DataFrame:
    """Return the n rows with the highest values in a numeric column, largest first."""
    _validate_numeric_column(df, column)

    if n < 1:
        raise ValueError("n must be at least 1")

    return df.nlargest(n, column)


def get_missing_value_percentages(df: pd.DataFrame) -> dict:
    """Return the percentage of missing values in each column."""
    if df.empty:
        return {column: 0.0 for column in df.columns}

    percentages = df.isnull().sum() / len(df) * 100
    return {column: round(float(value), 2) for column, value in percentages.items()}


def get_average_by_category(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> dict:
    """Return the average value for each category, rounded to 2 decimals."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    averages = df.groupby(category_column)[value_column].mean()
    return {category: round(float(value), 2) for category, value in averages.items()}


def get_share_above_threshold(df: pd.DataFrame, column: str, threshold: float) -> float:
    """Return the percentage of non-missing values that are greater than the threshold."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        return 0.0

    return round(float((values > threshold).sum() / len(values) * 100), 2)


def get_column_ratios(df: pd.DataFrame, numerator: str, denominator: str) -> list:
    """Return numerator / denominator for each row, or None where the denominator is 0 or missing."""
    _validate_numeric_column(df, numerator)
    _validate_numeric_column(df, denominator)

    ratios = []
    for top, bottom in zip(df[numerator], df[denominator]):
        if pd.isna(top) or pd.isna(bottom) or bottom == 0:
            ratios.append(None)
        else:
            ratios.append(round(float(top / bottom), 4))
    return ratios


def get_mode(df: pd.DataFrame, column: str):
    """Return the most frequent non-missing value in a column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    counts = df[column].dropna().value_counts()

    if counts.empty:
        raise ValueError(f"Column has no values: {column}")

    return counts.index[0]


def get_growth_rate(df: pd.DataFrame, column: str) -> float:
    """Return the percentage change from the first to the last non-missing value."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if len(values) < 2:
        raise ValueError(f"Need at least two values to compute growth: {column}")

    first = values.iloc[0]
    last = values.iloc[-1]

    if first == 0:
        raise ValueError(f"Cannot compute growth from a zero starting value: {column}")

    return round(float((last - first) / first * 100), 2)


def get_rows_in_range(
    df: pd.DataFrame,
    column: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    """Return rows where the column value is between low and high (inclusive)."""
    _validate_numeric_column(df, column)

    if low > high:
        raise ValueError("low must not be greater than high")

    return df[(df[column] >= low) & (df[column] <= high)]


def normalize_column(df: pd.DataFrame, column: str) -> list:
    """Scale a numeric column to the 0-1 range using min-max normalization."""
    _validate_numeric_column(df, column)

    minimum = df[column].min()
    maximum = df[column].max()

    if maximum == minimum:
        return [None if pd.isna(value) else 0.0 for value in df[column]]

    scaled = (df[column] - minimum) / (maximum - minimum)
    return [None if pd.isna(value) else round(float(value), 4) for value in scaled]


def fill_missing_with_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of the DataFrame with missing values in the column replaced by its median.

    The input DataFrame is not modified.
    """
    _validate_numeric_column(df, column)

    median = df[column].median()
    result = df.copy()
    result[column] = result[column].fillna(median)
    return result


def get_ranks(df: pd.DataFrame, column: str) -> list:
    """Return the rank of each value, where 1 is the highest. Ties share the best rank."""
    _validate_numeric_column(df, column)

    ranks = df[column].rank(ascending=False, method="min")
    return [None if pd.isna(rank) else int(rank) for rank in ranks]


def get_latest_value(df: pd.DataFrame, date_column: str, value_column: str):
    """Return the value from the row with the most recent date."""
    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")
    if value_column not in df.columns:
        raise KeyError(f"Column not found: {value_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")

    if dates.isna().all():
        raise ValueError(f"Column has no valid dates: {date_column}")

    return df.loc[dates.idxmax(), value_column]
