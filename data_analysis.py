
import pandas as pd


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """Return basic data-quality metrics for a DataFrame."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_value_count": int(df.isnull().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def get_weighted_average(df: pd.DataFrame, value_column: str, weight_column: str) -> float:
    """Return the average of value_column weighted by weight_column."""
    _validate_numeric_column(df, value_column)
    _validate_numeric_column(df, weight_column)

    pairs = df[[value_column, weight_column]].dropna()

    if pairs.empty or pairs[weight_column].sum() == 0:
        raise ValueError("Weights must have a non-zero total")

    weighted_total = (pairs[value_column] * pairs[weight_column]).sum()
    return round(float(weighted_total / pairs[weight_column].sum()), 4)


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


def get_category_share(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> dict:
    """Return each category's total as a percentage of the overall total, rounded to 2 decimals."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    overall_total = df[value_column].sum()

    if overall_total == 0:
        raise ValueError(f"Column total must not be 0: {value_column}")

    category_values = df.groupby(category_column)[value_column].sum()
    return {
        category: round(float(value / overall_total * 100), 2)
        for category, value in category_values.items()
    }


def get_cumulative_sum(df: pd.DataFrame, column: str) -> list:
    """Return the running total of a numeric column. Missing values are treated as 0."""
    _validate_numeric_column(df, column)

    running = df[column].fillna(0).cumsum()
    return [round(float(value), 2) for value in running]


def get_max_drop(df: pd.DataFrame, column: str) -> float:
    """Return the largest fall from an earlier peak to a later value, ignoring missing values.

    Returns 0.0 if the values never fall.
    """
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    drops = values.cummax() - values
    return round(float(max(drops.max(), 0) if len(values) > 1 else 0), 2)


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


def get_first_value(df: pd.DataFrame, column: str):
    """Return the first non-missing value in a column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    first = values.iloc[0]
    return first.item() if hasattr(first, "item") else first


def get_period_changes(df: pd.DataFrame, column: str) -> list:
    """Return the percentage change from each row to the next. The first row is None."""
    _validate_numeric_column(df, column)

    changes = df[column].pct_change(fill_method=None) * 100
    return [None if pd.isna(value) else round(float(value), 2) for value in changes]


def get_recent_average(df: pd.DataFrame, column: str, n: int = 3) -> float:
    """Return the average of the last n non-missing values, rounded to 2 decimals."""
    _validate_numeric_column(df, column)

    if n < 1:
        raise ValueError("n must be at least 1")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    recent = values.tail(n)
    return round(float(recent.mean()), 2)


def get_cumulative_percentage(df: pd.DataFrame, column: str) -> list:
    """Return the running total of a column as a percentage of the grand total, rounded to 2 decimals.

    Missing values count as 0.
    """
    _validate_numeric_column(df, column)
    values = df[column].fillna(0)
    grand_total = values.sum()

    if grand_total == 0:
        raise ValueError(f"Column total must not be 0: {column}")

    running = values.cumsum() / grand_total * 100
    return [round(float(value), 2) for value in running]


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


def get_monthly_counts(df: pd.DataFrame, date_column: str) -> dict:
    """Return the number of rows per month, keyed by "YYYY-MM". Rows with invalid dates are ignored."""
    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    keys = dates.dt.strftime("%Y-%m")
    counts = keys.value_counts().sort_index()

    return {key: int(count) for key, count in counts.items()}


def filter_by_category(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Return rows whose column matches any of the given values (case-insensitive)."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    wanted = {str(value).lower() for value in values}
    return df[df[column].astype(str).str.lower().isin(wanted)]


def get_expanding_average(df: pd.DataFrame, column: str) -> list:
    """Return the average of all values up to each row, rounded to 2 decimals.

    Missing values are skipped and returned as None.
    """
    _validate_numeric_column(df, column)
    values = df[column]
    running = values.expanding().mean()

    return [
        None if pd.isna(value) else round(float(average), 2)
        for value, average in zip(values, running)
    ]


def get_rows_above_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return the rows whose value is strictly greater than the column mean."""
    _validate_numeric_column(df, column)

    column_mean = df[column].mean()
    return df[df[column] > column_mean]


def get_percentile(df: pd.DataFrame, column: str, percentile: float) -> float:
    """Return the given percentile (0 to 100, inclusive) of a numeric column."""
    _validate_numeric_column(df, column)

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return float(values.quantile(percentile / 100))


def get_mean_absolute_deviation(df: pd.DataFrame, column: str) -> float:
    """Return the average distance of each value from the column mean, rounded to 2 decimals."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    center = values.mean()
    return round(float((values - center).abs().mean()), 2)


def get_date_range(df: pd.DataFrame, date_column: str) -> dict:
    """Return the earliest and latest valid dates in a column as "YYYY-MM-DD" strings."""
    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()

    if dates.empty:
        raise ValueError(f"Column has no valid dates: {date_column}")

    return {
        "start": dates.min().strftime("%Y-%m-%d"),
        "end": dates.max().strftime("%Y-%m-%d"),
    }


def get_percentage_of_total(df: pd.DataFrame, column: str) -> list:
    """Return each value as a percentage of the column total."""
    _validate_numeric_column(df, column)

    total = df[column].sum()

    if not total:
        return [None if pd.isna(value) else 0.0 for value in df[column]]

    shares = df[column] / total * 100
    return [None if pd.isna(value) else round(float(value), 2) for value in shares]


def get_quartiles(df: pd.DataFrame, column: str) -> dict:
    """Return the 25th, 50th and 75th percentiles of a numeric column."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no numeric values: {column}")

    return {
        "q1": float(values.quantile(0.25)),
        "q2": float(values.quantile(0.5)),
        "q3": float(values.quantile(0.75)),
    }


def get_range_by_category(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> dict:
    """Return the difference between the highest and lowest value for each category."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    grouped = df.groupby(category_column)[value_column]
    ranges = grouped.max() - grouped.min()
    return {category: round(float(value), 2) for category, value in ranges.items()}


def drop_duplicate_rows(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Return a copy without duplicate rows, keeping the first occurrence of each."""
    if subset:
        missing = [column for column in subset if column not in df.columns]
        if missing:
            raise KeyError(f"Column not found: {missing[0]}")

    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def get_weekend_share(df: pd.DataFrame, date_column: str, value_column: str) -> float:
    """Return the percentage of the total value that falls on Saturdays and Sundays, rounded to 2 decimals.

    Rows with invalid dates are ignored.
    """
    _validate_numeric_column(df, value_column)

    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid = dates.notna()
    total = df.loc[valid, value_column].sum()

    if total == 0:
        raise ValueError(f"Column total must not be 0: {value_column}")

    weekend = valid & (dates.dt.dayofweek >= 5)
    return round(float(df.loc[weekend, value_column].sum() / total * 100), 2)


def get_quarterly_totals(df: pd.DataFrame, date_column: str, value_column: str) -> dict:
    """Return the total value per quarter, keyed by "YYYY-Qn". Rows with invalid dates are ignored."""
    _validate_numeric_column(df, value_column)

    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid = dates.notna()
    dates = dates[valid]
    quarter_numbers = (dates.dt.month - 1) // 3 + 1
    keys = dates.dt.year.astype(str) + "-Q" + quarter_numbers.astype(str)
    totals = df.loc[valid, value_column].groupby(keys).sum()

    return {key: round(float(total), 2) for key, total in totals.items()}


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


def count_missing_by_row(df: pd.DataFrame) -> list:
    """Return the number of missing values in each row."""
    return [int(count) for count in df.isnull().sum(axis=1)]


def get_z_scores(df: pd.DataFrame, column: str) -> list:
    """Return the z-score of each value: (value - mean) / standard deviation."""
    _validate_numeric_column(df, column)

    mean = df[column].mean()
    scale = df[column].std()

    if not scale:
        return [None if pd.isna(value) else 0.0 for value in df[column]]

    scores = (df[column] - mean) / scale
    return [None if pd.isna(value) else round(float(value), 4) for value in scores]


def get_row_totals(df: pd.DataFrame, columns: list) -> list:
    """Return the sum of the given numeric columns for each row. Missing values count as 0."""
    for column in columns:
        _validate_numeric_column(df, column)

    totals = df[columns].fillna(0).max(axis=1)
    return [round(float(value), 2) for value in totals]


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


def get_days_between(df: pd.DataFrame, start_column: str, end_column: str) -> list:
    """Return the number of days from the start date to the end date for each row."""
    for column in (start_column, end_column):
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")

    start = pd.to_datetime(df[start_column], errors="coerce")
    end = pd.to_datetime(df[end_column], errors="coerce")
    days = (end - start).dt.days

    return [None if pd.isna(value) else int(value) for value in days]


def get_percent_of_max(df: pd.DataFrame, column: str) -> list:
    """Return each value as a percentage of the column maximum, rounded to 2 decimals.

    Missing values are returned as None.
    """
    _validate_numeric_column(df, column)
    reference = df[column].max()

    if reference == 0:
        raise ValueError(f"Column maximum must not be 0: {column}")

    percents = df[column] / reference * 100
    return [None if pd.isna(value) else round(float(value), 2) for value in percents]


def get_monthly_totals(df: pd.DataFrame, date_column: str, value_column: str) -> dict:
    """Return the total value per month, keyed by "YYYY-MM"."""
    _validate_numeric_column(df, value_column)

    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    months = dates.dt.strftime("%Y-%m")
    totals = df[value_column].groupby(months).sum()

    return {month: float(total) for month, total in totals.items()}


def get_best_average_category(df: pd.DataFrame, category_column: str, value_column: str):
    """Return the category with the highest average value, ignoring missing values."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    values = df[[category_column, value_column]].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {value_column}")

    averages = values.groupby(category_column)[value_column].mean()
    return averages.idxmax()


def count_values_above(df: pd.DataFrame, column: str, threshold: float) -> int:
    """Return how many values are strictly greater than the threshold."""
    _validate_numeric_column(df, column)

    return int((df[column] > threshold).sum())


def get_unique_count(df: pd.DataFrame, column: str) -> int:
    """Return the number of distinct non-missing values in a column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    return int(df[column].nunique())


def count_duplicate_rows(df: pd.DataFrame, subset: list = None) -> int:
    """Return the number of rows that repeat an earlier row.

    When subset is given, only those columns are compared.
    """
    if subset is not None:
        for column in subset:
            if column not in df.columns:
                raise KeyError(f"Column not found: {column}")

    return int(df.duplicated(subset=subset, keep="first").sum())


def get_top_n_rows(df: pd.DataFrame, column: str, n: int = 5) -> pd.DataFrame:
    """Return the n rows with the highest values in a numeric column, largest first."""
    _validate_numeric_column(df, column)

    if n < 1:
        raise ValueError("n must be at least 1")

    return df.nlargest(n, column)


def filter_rows_containing(df: pd.DataFrame, column: str, text: str) -> pd.DataFrame:
    """Return the rows whose column contains the given text, ignoring upper/lower case.

    The text is matched literally. Missing values never match.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    matches = df[column].astype("string").str.contains(text, case=False, na=False, regex=False)
    return df[matches]


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


def filter_rows_between_dates(
    df: pd.DataFrame,
    date_column: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return the rows whose date is between start and end, including both.

    Rows with invalid dates are excluded.
    """
    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    if start_date > end_date:
        raise ValueError("start must not be after end")

    dates = pd.to_datetime(df[date_column], errors="coerce")
    return df[(dates >= start_date) & (dates <= end_date)]


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


def get_word_counts(df: pd.DataFrame, column: str) -> list:
    """Return the number of words in each value of a text column.

    Missing values are returned as None.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    return [None if pd.isna(value) else len(str(value).split()) for value in df[column]]


def get_longest_streak_above(df: pd.DataFrame, column: str, threshold: float) -> int:
    """Return the length of the longest run of consecutive values greater than threshold.

    Missing values end a run.
    """
    _validate_numeric_column(df, column)

    longest = 0
    streak = 0
    for value in df[column]:
        if not pd.isna(value) and value > threshold:
            streak += 1
        else:
            longest = max(longest, streak)
            streak = 0

    return max(longest, streak)


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


def get_days_since_first(df: pd.DataFrame, date_column: str) -> list:
    """Return the number of days between each date and the earliest valid date.

    Invalid dates are returned as None.
    """
    if date_column not in df.columns:
        raise KeyError(f"Column not found: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="coerce")

    if dates.notna().sum() == 0:
        raise ValueError(f"Column has no valid dates: {date_column}")

    first = dates.min()
    days = (dates - first).dt.days
    return [None if pd.isna(value) else int(value) for value in days]


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


def get_true_percentage(df: pd.DataFrame, column: str) -> float:
    """Return the percentage of non-missing values that are True, rounded to 2 decimals."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    values = df[column].dropna()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    return round(float(values.astype(bool).sum() / len(values) * 100), 2)


def fill_missing_with_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of the DataFrame with missing values in the column replaced by its median.

    The input DataFrame is not modified.
    """
    _validate_numeric_column(df, column)

    median = df[column].median()
    result = df.copy()
    result[column] = result[column].fillna(median)
    return result


def get_second_highest_value(df: pd.DataFrame, column: str) -> float:
    """Return the second highest distinct value in a numeric column, ignoring missing values."""
    _validate_numeric_column(df, column)
    values = df[column].dropna()

    if values.nunique() < 2:
        raise ValueError(f"Column needs at least two distinct values: {column}")

    return float(values.drop_duplicates().nlargest(2).iloc[-1])


def get_ranks(df: pd.DataFrame, column: str) -> list:
    """Return the rank of each value, where 1 is the highest. Ties share the best rank."""
    _validate_numeric_column(df, column)

    ranks = df[column].rank(ascending=False, method="min")
    return [None if pd.isna(rank) else int(rank) for rank in ranks]


def get_trimmed_mean(df: pd.DataFrame, column: str, proportion: float = 0.1) -> float:
    """Return the mean after dropping the given proportion of values from each end, rounded to 2 decimals."""
    _validate_numeric_column(df, column)

    if not 0 <= proportion < 0.5:
        raise ValueError("proportion must be at least 0 and less than 0.5")

    values = df[column].dropna().sort_values()

    if values.empty:
        raise ValueError(f"Column has no values: {column}")

    cut = int(len(values) * proportion)
    trimmed = values.iloc[cut : len(values) - cut]
    return round(float(trimmed.mean()), 2)


def get_top_categories(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    n: int = 3,
) -> list:
    """Return the n categories with the highest total value, highest first."""
    _validate_numeric_column(df, value_column)

    if category_column not in df.columns:
        raise KeyError(f"Column not found: {category_column}")

    if n < 1:
        raise ValueError("n must be at least 1")

    category_sums = df.groupby(category_column)[value_column].sum()
    return list(category_sums.sort_values().head(n).index)


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
