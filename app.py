import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import re
import json
from io import StringIO

from io import BytesIO
from datetime import datetime

import plotly.io as pio

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
    KeepTogether,
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "❌ GEMINI_API_KEY not found. "
        "Please check your .env file."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemma-4-26b-a4b-it"

st.set_page_config(
    page_title="InsightForge",
    page_icon="🔎",
    layout="wide"
)

# =========================================================
# ASK YOUR DATA — SAFE, PYTHON-EXECUTED ANALYSIS
# =========================================================

def _json_safe(value):
    """Convert common Pandas/NumPy values to JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _column_context(df):
    """Small schema/sample context for planning; never sends the full dataset."""
    columns = []
    for col in df.columns:
        series = df[col]
        examples = [_json_safe(v) for v in series.dropna().head(3).tolist()]
        columns.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "examples": examples,
            "unique_count": int(series.nunique(dropna=True)),
        })
    sample = df.head(6).copy()
    sample.columns = [str(c) for c in sample.columns]
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "sample_rows": _json_safe(sample.to_dict(orient="records")),
    }


def _extract_json_object(text):
    """Parse a JSON object even if the model wraps it in a code fence."""
    if not text:
        raise ValueError("The AI returned an empty analysis plan.")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The AI could not produce a valid analysis plan. Please rephrase the question.")


def _execute_analysis_plan(df, plan):
    """Execute only allow-listed Pandas operations; never execute generated code."""
    operation = str(plan.get("operation", "")).strip().lower()
    columns = list(df.columns)
    column_lookup = {str(c): c for c in columns}

    def get_column(key, required=True):
        name = plan.get(key)
        if name is None and not required:
            return None
        if not isinstance(name, str) or name not in column_lookup:
            raise ValueError(f"The requested {key.replace('_', ' ')} is not a column in this dataset.")
        return column_lookup[name]

    limit = plan.get("limit", 10)
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 10

    if operation == "dataset_summary":
        numeric = df.select_dtypes(include=np.number)
        result = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": [str(c) for c in numeric.columns],
            "categorical_columns": [str(c) for c in df.columns if c not in numeric.columns],
        }
    elif operation == "column_stats":
        col = get_column("column")
        series = df[col]
        result = {"column": str(col), "dtype": str(series.dtype), "count_non_null": int(series.notna().sum()),
                  "missing": int(series.isna().sum()), "unique_values": int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            result.update({"mean": _json_safe(clean.mean()), "median": _json_safe(clean.median()),
                           "min": _json_safe(clean.min()), "max": _json_safe(clean.max()),
                           "sum": _json_safe(clean.sum()), "std_dev": _json_safe(clean.std())})
        else:
            result["most_common_values"] = _json_safe(series.value_counts(dropna=False).head(limit).to_dict())
    elif operation == "missing_values":
        counts = df.isna().sum()
        result = {"total_missing_cells": int(counts.sum()), "by_column": {str(k): int(v) for k, v in counts.items()},
                  "columns_with_missing_values": {str(k): int(v) for k, v in counts.items() if v > 0}}
    elif operation == "duplicates":
        result = {"duplicate_rows": int(df.duplicated().sum()), "total_rows": int(len(df)),
                  "duplicate_percentage": round(float(df.duplicated().mean() * 100), 2) if len(df) else 0.0}
    elif operation == "value_counts":
        col = get_column("column")
        result = {"column": str(col), "counts": _json_safe(df[col].value_counts(dropna=False).head(limit).to_dict()),
                  "total_distinct_non_null": int(df[col].nunique(dropna=True))}
    elif operation == "aggregate":
        group_col = get_column("group_by")
        value_col = get_column("value_column")
        agg = str(plan.get("aggregation", "sum")).lower()
        if agg not in {"sum", "mean", "median", "min", "max", "count"}:
            raise ValueError("Unsupported aggregation. Try sum, mean, median, min, max, or count.")
        if agg != "count" and not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError(f"'{value_col}' is not numeric, so {agg} cannot be calculated.")
        grouped = df.groupby(group_col, dropna=False)[value_col]
        aggregated = getattr(grouped, agg)().sort_values(ascending=False, na_position="last")
        result = {"group_by": str(group_col), "value_column": str(value_col), "aggregation": agg,
                  "groups_returned": int(len(aggregated)),
                  "results": [{"group": _json_safe(k), "value": _json_safe(v)} for k, v in aggregated.head(limit).items()]}
    elif operation == "correlation":
        col1, col2 = get_column("column_1"), get_column("column_2")
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
            raise ValueError("Correlation requires two numeric columns.")
        valid = df[[col1, col2]].dropna()
        if len(valid) < 2:
            raise ValueError("There are not enough complete rows to calculate correlation.")
        result = {"column_1": str(col1), "column_2": str(col2), "complete_rows": int(len(valid)),
                  "pearson_correlation": _json_safe(valid[col1].corr(valid[col2]))}
    elif operation == "trend":
        date_col, value_col = get_column("date_column"), get_column("value_column")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError(f"'{value_col}' must be numeric to calculate a trend.")
        dates = pd.to_datetime(df[date_col], errors="coerce")
        work = pd.DataFrame({"date": dates, "value": pd.to_numeric(df[value_col], errors="coerce")}).dropna()
        work = work.sort_values("date")
        if len(work) < 2:
            raise ValueError("At least two valid date/value rows are needed for trend analysis.")
        first, last = float(work["value"].iloc[0]), float(work["value"].iloc[-1])
        change_pct = ((last - first) / abs(first) * 100) if first != 0 else None
        result = {"date_column": str(date_col), "value_column": str(value_col), "valid_points": int(len(work)),
                  "first_date": work["date"].iloc[0].isoformat(), "first_value": first,
                  "last_date": work["date"].iloc[-1].isoformat(), "last_value": last,
                  "absolute_change": last - first, "percentage_change": _json_safe(change_pct),
                  "direction": "increased" if last > first else "decreased" if last < first else "unchanged"}
    elif operation == "top_rows":
        sort_col = get_column("sort_by")
        order = str(plan.get("order", "descending")).lower()
        ascending = order in {"ascending", "asc", "smallest", "lowest"}
        sorted_df = df.sort_values(sort_col, ascending=ascending, na_position="last").head(limit)
        result = {"sorted_by": str(sort_col), "order": "ascending" if ascending else "descending",
                  "rows": _json_safe(sorted_df.astype(object).where(pd.notna(sorted_df), None).to_dict(orient="records"))}
    elif operation == "extreme_row":
        value_col = get_column("value_column")
        direction = str(plan.get("direction", "min")).strip().lower()

        if direction not in {"min", "max"}:
            raise ValueError("Extreme direction must be 'min' or 'max'.")

        values = pd.to_numeric(df[value_col], errors="coerce")
        valid = values.dropna()

        if valid.empty:
            raise ValueError(
                f"'{value_col}' has no usable numeric values to compare."
            )

        extreme_value = valid.min() if direction == "min" else valid.max()

        # Return every row tied for the minimum or maximum.
        matching = df.loc[values.eq(extreme_value)].copy()

        result = {
            "operation": "extreme_row",
            "value_column": str(value_col),
            "direction": direction,
            "extreme_value": _json_safe(extreme_value),
            "matching_rows": int(len(matching)),
            "rows": _json_safe(
                matching.astype(object)
                .where(pd.notna(matching), None)
                .head(limit)
                .to_dict(orient="records")
            ),
        }
    else:
        raise ValueError("I couldn't map that question to a supported analysis. Try asking about summary, statistics, groups, trends, missing values, duplicates, or correlation.")
    return _json_safe(result)


def _create_analysis_plan(question, df):
    """Use fast local rules for common questions; call Gemma only as a fallback."""
    q = question.lower().strip()
    columns = list(df.columns)
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = [c for c in columns if c not in numeric]

    # Match real column names first so questions can use mixed-case headers.
    mentioned = {str(c).lower(): str(c) for c in columns}
    mentioned_cols = [original for lowered, original in mentioned.items() if lowered in q]

    if any(term in q for term in ("missing", "null", "empty values", "incomplete")):
        return {"operation": "missing_values"}
    if any(term in q for term in ("duplicate", "duplicated")):
        return {"operation": "duplicates"}
    if any(term in q for term in ("summarize", "summary", "dataset overview", "how many rows", "how many columns")):
        return {"operation": "dataset_summary"}

    if any(term in q for term in ("average", "mean", "median", "maximum", "minimum", "max value", "min value", "standard deviation", "sum of")):
        col = mentioned_cols[0] if mentioned_cols else (numeric[0] if numeric else None)
        if col is not None:
            return {"operation": "column_stats", "column": col}

    if any(term in q for term in ("highest total", "largest total", "most sales", "highest sales", "top product", "best selling", "lowest total", "least sales", "lowest sales")) and numeric and categorical:
        group_col = next((c for c in mentioned_cols if c in categorical), categorical[0])
        value_col = next((c for c in mentioned_cols if c in numeric), numeric[0])
        agg = "mean" if any(term in q for term in ("average", "mean")) else "sum"
        return {"operation": "aggregate", "group_by": group_col, "value_column": value_col,
                "aggregation": agg, "limit": 10}

    if any(term in q for term in ("most common", "frequency", "value counts", "count of each", "how many different")):
        col = mentioned_cols[0] if mentioned_cols else (categorical[0] if categorical else (columns[0] if columns else None))
        if col is not None:
            return {"operation": "value_counts", "column": col, "limit": 10}

    # Single lowest/highest record (e.g. "lowest sale and when").
    extreme_words = re.search(
        r"\b(lowest|smallest|minimum|min|least|highest|largest|biggest|maximum|max)\b", q
    )
    group_words = re.search(
        r"\b(total|average|mean|median|by|per|each|category|region|product|group)\b", q
    )
    if extreme_words and not group_words and numeric:
        direction = "min" if extreme_words.group(1) in (
            "lowest", "smallest", "minimum", "min", "least"
        ) else "max"
        value_col = next((c for c in mentioned_cols if c in numeric), None)
        if value_col is None:
            hints = ("sale", "amount", "revenue", "total", "price", "value")
            value_col = next(
                (c for c in numeric if any(h in str(c).lower() for h in hints)),
                numeric[0],
            )
        return {"operation": "extreme_row", "value_column": value_col,
                "direction": direction, "limit": 5}

    # Gemma fallback for less common or ambiguous questions.
    context = _column_context(df)
    planner_prompt = f"""You are the analysis planner for InsightForge. Convert the user's question into ONE safe JSON analysis plan. Do not answer the question. Do not write code.

Allowed operations and required fields:
- dataset_summary: {{"operation":"dataset_summary"}}
- column_stats: {{"operation":"column_stats","column":"exact column name"}}
- missing_values: {{"operation":"missing_values"}}
- duplicates: {{"operation":"duplicates"}}
- value_counts: {{"operation":"value_counts","column":"exact column name","limit":10}}
- aggregate: {{"operation":"aggregate","group_by":"categorical/group column","value_column":"column to measure","aggregation":"sum|mean|median|min|max|count","limit":10}}
- correlation: {{"operation":"correlation","column_1":"numeric column","column_2":"numeric column"}}
- trend: {{"operation":"trend","date_column":"date column","value_column":"numeric column"}}
- top_rows: {{"operation":"top_rows","sort_by":"column","order":"descending|ascending","limit":10}}
- extreme_row: {{"operation":"extreme_row","value_column":"numeric column","direction":"min|max","limit":5}}

Use only exact column names from the dataset context. If the user asks which category/region/product has the highest or lowest value, use aggregate with the appropriate value column and sum/mean as the wording implies. For "lowest/highest single value and when/which row", use extreme_row. If they ask for most frequent categories, use value_counts. For a general question, choose dataset_summary. Return ONLY a JSON object. Dataset context: {json.dumps(context, ensure_ascii=False, default=str)}
User question: {question}
JSON plan:"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=planner_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    text = response.text
    if not text and response.candidates:
        parts = response.candidates[0].content.parts or []
        text = "".join(part.text for part in parts if getattr(part, "text", None))
    return _extract_json_object(text)


def _local_evidence_answer(question, evidence):
    
    """Fast, deterministic answer for common questions using Python-computed evidence."""
    if not isinstance(evidence, dict):
        
        return "Python completed the analysis, but the result was not in a format I can summarize."
    if evidence.get("operation") == "extreme_row":
        value_col = evidence.get("value_column", "value")
        direction = evidence.get("direction", "min")
        value = evidence.get("extreme_value")
        rows = evidence.get("rows", [])

        label = "minimum" if direction == "min" else "maximum"

        if not rows:
            return (
                f"Python calculated the {label} for **{value_col}**, "
                "but no matching rows were available to display."
            )

        row_details = []
        for row in rows[:5]:
            details = ", ".join(
                f"**{column}**: {item}"
                for column, item in row.items()
            )
            row_details.append(details)

        answer = (
            f"The {label} **{value_col}** is **{value}**. "
            f"Python found {evidence.get('matching_rows', len(rows))} "
            "matching row(s)."
        )

        if row_details:
            answer += "\n\nMatching row details:\n\n" + "\n\n".join(
                f"- {details}" for details in row_details
            )

        return answer
    results = evidence.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict) and "group" in first and "value" in first:
            group_col = evidence.get("group_by", "group")
            value_col = evidence.get("value_column", "value")
            aggregation = evidence.get("aggregation", "sum")
            winner = first.get("group")
            value = first.get("value")
            action = {
                "sum": "highest total",
                "mean": "highest average",
                "median": "highest median",
                "max": "highest value",
                "min": "highest minimum",
                "count": "highest count",
            }.get(aggregation, f"highest {aggregation}")
            rest = results[1:4]
            comparison = ""
            if rest:
                comparison = " Other results include " + ", ".join(
                    f"{item.get('group')}: {item.get('value'):,}" if isinstance(item.get("value"), (int, float)) else f"{item.get('group')}: {item.get('value')}"
                    for item in rest
                ) + "."
            if isinstance(value, (int, float)):
                value_text = f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
            else:
                value_text = str(value)
            return (
                f"**{winner}** has the {action} {value_col}, at **{value_text}** "
                f"(grouped by {group_col}).{comparison} "
                "These values were calculated by Python from the uploaded dataset."
            )

    if "mean" in evidence and "column" in evidence:
        col = evidence.get("column", "The selected column")
        mean = evidence.get("mean")
        if isinstance(mean, (int, float)):
            return f"The average **{col}** is **{mean:,.2f}**. Python calculated this using {evidence.get('count_non_null', 'the available')} non-missing values."
        return f"I couldn't calculate an average for **{col}** because it does not contain usable numeric values."

    if "total_missing_cells" in evidence:
        total = evidence["total_missing_cells"]
        if total == 0:
            return "There are **no missing values** in the uploaded dataset. Python checked every column."
        details = evidence.get("columns_with_missing_values", {})
        breakdown = ", ".join(f"{col}: {count}" for col, count in details.items())
        return f"The dataset contains **{total:,} missing cells**. " + (f"By column: {breakdown}." if breakdown else "")

    if "duplicate_rows" in evidence and "total_rows" in evidence:
        return f"Python found **{evidence['duplicate_rows']:,} duplicate rows** out of {evidence['total_rows']:,} total rows ({evidence.get('duplicate_percentage', 0)}%)."

    if "row_count" in evidence and "column_count" in evidence:
        return (
            f"The dataset contains **{evidence['row_count']:,} rows** and "
            f"**{evidence['column_count']:,} columns**. It has "
            f"**{evidence.get('missing_cells', 0):,} missing cells** and "
            f"**{evidence.get('duplicate_rows', 0):,} duplicate rows**."
        )

    if "rows" in evidence and "columns" in evidence and "column_names" in evidence:
        return (
            f"The dataset contains **{evidence['rows']:,} rows** and **{evidence['columns']:,} columns**. "
            f"Missing cells: **{evidence.get('missing_cells', 0):,}**. "
            f"Duplicate rows: **{evidence.get('duplicate_rows', 0):,}**. "
            f"Columns: {', '.join(map(str, evidence.get('column_names', [])))}."
        )

    if "pearson_correlation" in evidence:
        corr = evidence.get("pearson_correlation")
        return (
            f"The Pearson correlation between **{evidence.get('column_1')}** and "
            f"**{evidence.get('column_2')}** is **{corr:.3f}** based on "
            f"{evidence.get('complete_rows', 0)} complete rows. Correlation does not prove causation."
            if isinstance(corr, (int, float)) else
            "Python could not calculate a correlation from the available values."
        )

    if "percentage_change" in evidence and "direction" in evidence:
        pct = evidence.get("percentage_change")
        pct_text = f"{pct:.2f}%" if isinstance(pct, (int, float)) else "not available (the starting value is zero)"
        return (
            f"**{evidence.get('value_column')} {evidence['direction']}** from "
            f"{evidence.get('first_value')} on {evidence.get('first_date')} to "
            f"{evidence.get('last_value')} on {evidence.get('last_date')}. "
            f"Percentage change: **{pct_text}** across {evidence.get('valid_points', 0)} valid data points."
        )

    if "counts" in evidence and "column" in evidence:
        counts = evidence.get("counts", {})
        summary = ", ".join(f"{key}: {value}" for key, value in list(counts.items())[:5])
        return f"Most frequent values in **{evidence['column']}**: {summary or 'No values available'}."

    return None


def _answer_data_question(question, computed_evidence):
    """Use instant Python-backed explanations for common questions; Gemma for complex ones."""
    local_answer = _local_evidence_answer(question, computed_evidence)
    if local_answer:
        return local_answer

    answer_prompt = f"""You are InsightForge's data analyst. Answer the user's question using ONLY the verified Python-computed evidence below. Explain the result in clear, concise language and mention key numbers. If the evidence does not fully answer the question, say what is missing. Do not invent facts, causes, or calculations. Clearly label any possible explanation as a hypothesis, not a proven fact. Do not claim correlation proves causation.

User question: {question}
Verified Python-computed evidence (JSON):
{json.dumps(computed_evidence, ensure_ascii=False, indent=2, default=str)}

Keep the answer concise (2–4 sentences). Include the key verified numbers. Do not add a follow-up question unless it is genuinely useful."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=answer_prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=180)
        )
        answer = (getattr(response, "text", None) or "").strip()
        if answer:
            return answer
    except Exception:
        pass

    # Never leave the user with an empty answer if the model returns no text.
    return (
        "Python completed the calculation and the verified evidence is shown below. "
        "Gemma did not return a text explanation this time; please use the evidence table for the exact result."
    )



# =========================================================
# CUSTOM STYLE — INSIGHTFORGE PREMIUM UI
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #687792;
        --line: #E2E8F4;
        --blue: #315EFB;
        --violet: #7956E8;
        --teal: #14B8A6;
        --paper: #F8FAFF;
        --surface: #FFFFFF;
    }

    /* Main application background */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 50% 8%,
                rgba(121, 86, 232, 0.09),
                transparent 42%
            ),
            radial-gradient(
                ellipse at 10% 30%,
                rgba(49, 94, 251, 0.045),
                transparent 35%
            ),
            var(--paper);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 255, 0.88);
    }

    /* Main content spacing */
    .main .block-container {
        max-width: 1440px;
        padding: 2rem 2.5rem 4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        letter-spacing: -0.5px;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #475570;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        letter-spacing: -0.8px;
    }

    h1 {
        font-weight: 800;
    }

    h2 {
        font-size: 25px !important;
        font-weight: 700;
    }

    h3 {
        font-size: 18px !important;
        font-weight: 700;
    }

    p, li {
        line-height: 1.75;
    }

    .main-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(34px, 4vw, 48px);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.15;
        color: var(--ink);
    }

    .subtitle {
        font-size: 16px;
        line-height: 1.8;
        color: var(--muted);
        max-width: 700px;
        margin-top: 12px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 5px 24px rgba(36, 54, 96, 0.04);
        transition: border-color 180ms ease,
                    box-shadow 180ms ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #C6D2FF;
        box-shadow: 0 8px 28px rgba(49, 94, 251, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 44px;
        border: 1px solid var(--line);
        border-radius: 11px;
        background: #FFFFFF;
        color: #263451;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        transition: all 180ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #EEF2FF;
        border-color: #A9BAFF;
        color: var(--blue);
        box-shadow: 0 5px 16px rgba(49, 94, 251, 0.08);
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(135deg, #315EFB, #7956E8);
        border: none;
        color: #FFFFFF;
        font-weight: 700;
        box-shadow: 0 5px 15px rgba(49, 94, 251, 0.15);
    }

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #244CE0, #6844D8);
        color: #FFFFFF;
        box-shadow: 0 7px 20px rgba(49, 94, 251, 0.22);
        transform: translateY(-1px);
    }

    /* File upload */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #A9BAF0;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 8px 28px rgba(49, 94, 251, 0.045);
    }

    [data-testid="stFileUploader"] section {
        background: #F9FAFF;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] button {
        border-radius: 9px;
        border-color: #D8E0F2;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] small {
        color: var(--muted);
    }

    /* Data tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 13px;
        overflow: hidden;
    }

    /* Text inputs, dropdowns and chat fields */
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #DCE3F0;
    }

    [data-baseweb="input"]:focus-within,
    [data-baseweb="textarea"]:focus-within {
        border-color: var(--blue);
        box-shadow: 0 0 0 2px rgba(49, 94, 251, 0.09);
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 14px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--line);
    }

    [data-baseweb="tab"] {
        height: 46px;
        padding: 0 16px;
        color: var(--muted);
        font-weight: 600;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: var(--blue);
    }

    /* Expanders and alerts */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 13px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dividers and captions */
    hr {
        border-color: var(--line);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    /* Keep the layout responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.2rem 1rem 3rem;
        }

        .main-title {
            letter-spacing: -1px;
        }

        [data-testid="stMetric"] {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =========================================================
# PDF REPORT GENERATOR — INSIGHTFORGE
# =========================================================

def build_pdf_report(
    dataset_name,
    df,
    numeric_columns,
    report_text="",
    trend_chart=None,
    histogram_chart=None,
    anomaly_df=None,
):
    """Build a professional, evidence-based PDF report in memory."""

    import html
    import re
    from io import BytesIO
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        KeepTogether,
        Image as PDFImage,
        LongTable,
    )

    import pandas as pd

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"InsightForge Report - {dataset_name}",
        author="InsightForge",
    )

    # -----------------------------------------------------
    # REPORT STYLES
    # -----------------------------------------------------

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="IFTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=31,
        textColor=colors.HexColor("#17212F"),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="IFSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="IFSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#315EFB"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="IFSubheading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#17212F"),
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="IFBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
        splitLongWords=1,
    ))

    styles.add(ParagraphStyle(
        name="IFSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#475569"),
    ))

    styles.add(ParagraphStyle(
        name="IFCoverMetric",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17212F"),
    ))

    styles.add(ParagraphStyle(
        name="IFTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#17212F"),
    ))

    story = []

    # -----------------------------------------------------
    # SAFE TEXT AND MARKDOWN HELPERS
    # -----------------------------------------------------

    def safe_text(value):
        """Escape arbitrary dataset or AI text for ReportLab."""
        if value is None:
            return ""
        return html.escape(str(value), quote=False)

    def inline_markdown(value):
        """Render common Markdown emphasis safely."""
        value = safe_text(value)

        # Inline code
        value = re.sub(
            r"`([^`]+)`",
            r'<font name="Courier">\1</font>',
            value,
        )

        # Bold
        value = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            value,
        )

        # Italic
        value = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            r"<i>\1</i>",
            value,
        )

        return value

    def fmt_number(value):
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)

    def table_style(header=True, font_size=7):
        commands = [
            (
                "GRID", (0, 0), (-1, -1),
                0.4, colors.HexColor("#DDE4EE"),
            ),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            (
                "ROWBACKGROUNDS", (0, 1), (-1, -1),
                [colors.white, colors.HexColor("#F7F9FC")],
            ),
        ]

        if header:
            commands.extend([
                (
                    "BACKGROUND", (0, 0), (-1, 0),
                    colors.HexColor("#EEF2FF"),
                ),
                (
                    "FONTNAME", (0, 0), (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "TEXTCOLOR", (0, 0), (-1, 0),
                    colors.HexColor("#17212F"),
                ),
                ("REPEATROWS", (0, 0), (-1, 0)),
            ])

        return TableStyle(commands)

    # -----------------------------------------------------
    # COVER / DATASET OVERVIEW
    # -----------------------------------------------------

    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("INSIGHTFORGE", styles["IFTitle"]))
    story.append(Paragraph(
        "AI DATA DETECTIVE  /  DATA INVESTIGATION REPORT",
        styles["IFSubtitle"],
    ))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph(
        f"<b>Dataset:</b> {safe_text(dataset_name)}<br/>"
        f"<b>Generated:</b> "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}<br/>"
        f"<b>Report type:</b> Data quality, statistical analysis "
        f"and AI-assisted interpretation",
        styles["IFBody"],
    ))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Executive overview",
        styles["IFSection"],
    ))

    row_count = len(df)
    column_count = len(df.columns)
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    numeric_count = len(numeric_columns)

    metric_data = [
        ["RECORDS", "COLUMNS", "MISSING CELLS", "DUPLICATE ROWS"],
        [
            Paragraph(f"{row_count:,}", styles["IFCoverMetric"]),
            Paragraph(f"{column_count:,}", styles["IFCoverMetric"]),
            Paragraph(f"{missing_total:,}", styles["IFCoverMetric"]),
            Paragraph(f"{duplicate_total:,}", styles["IFCoverMetric"]),
        ],
    ]

    metric_table = Table(
        metric_data,
        colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm],
    )

    metric_table.setStyle(TableStyle([
        (
            "BACKGROUND", (0, 0), (-1, 0),
            colors.HexColor("#EEF2FF"),
        ),
        (
            "TEXTCOLOR", (0, 0), (-1, 0),
            colors.HexColor("#475569"),
        ),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#DDE4EE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE4EE")),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))

    story.append(metric_table)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        f"The dataset contains {row_count:,} records and "
        f"{column_count:,} columns, including "
        f"{numeric_count:,} numeric columns. "
        f"The initial quality check identified {missing_total:,} "
        f"missing cells and {duplicate_total:,} duplicate rows. "
        f"These counts describe the uploaded data and do not, by "
        f"themselves, establish the business impact of any issue.",
        styles["IFBody"],
    ))

    story.append(Paragraph(
        "Report contents",
        styles["IFSubheading"],
    ))

    story.append(Paragraph(
        "1. Data quality assessment<br/>"
        "2. Statistical evidence<br/>"
        "3. Trend and distribution charts, when available<br/>"
        "4. Detected anomalies, when available<br/>"
        "5. AI interpretation and recommendations, when supplied",
        styles["IFBody"],
    ))

    # -----------------------------------------------------
    # DATA QUALITY TABLE
    # -----------------------------------------------------

    story.append(Paragraph(
        "1. Data quality assessment",
        styles["IFSection"],
    ))

    quality_data = [[
        "Column", "Data type", "Missing", "Missing %", "Unique values"
    ]]

    for column in df.columns:
        missing = int(df[column].isna().sum())
        missing_pct = (
            missing / len(df) * 100
            if len(df) > 0 else 0
        )

        quality_data.append([
            Paragraph(safe_text(column), styles["IFSmall"]),
            Paragraph(safe_text(df[column].dtype), styles["IFSmall"]),
            f"{missing:,}",
            f"{missing_pct:.1f}%",
            f"{df[column].nunique():,}",
        ])

    quality_table = LongTable(
        quality_data,
        colWidths=[52 * mm, 34 * mm, 20 * mm, 25 * mm, 32 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    quality_table.setStyle(table_style(font_size=7))
    story.append(quality_table)

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Interpretation note: missing values, duplicates and unique "
        "counts are diagnostic indicators. Whether they require "
        "correction depends on the meaning of each field and the "
        "intended use of the dataset.",
        styles["IFBody"],
    ))

    # -----------------------------------------------------
    # NUMERIC STATISTICS
    # -----------------------------------------------------

    story.append(Paragraph(
        "2. Statistical evidence",
        styles["IFSection"],
    ))

    if numeric_columns:
        stats_data = [[
            "Column", "Valid count", "Mean",
            "Median", "Minimum", "Maximum",
        ]]

        for column in numeric_columns:
            values = pd.to_numeric(
                df[column], errors="coerce"
            ).dropna()

            if values.empty:
                continue

            stats_data.append([
                Paragraph(safe_text(column), styles["IFSmall"]),
                f"{len(values):,}",
                fmt_number(values.mean()),
                fmt_number(values.median()),
                fmt_number(values.min()),
                fmt_number(values.max()),
            ])

        if len(stats_data) > 1:
            stats_table = LongTable(
                stats_data,
                colWidths=[
                    38 * mm, 22 * mm, 28 * mm,
                    28 * mm, 28 * mm, 28 * mm,
                ],
                repeatRows=1,
                hAlign="LEFT",
            )
            stats_table.setStyle(table_style(font_size=6.5))
            story.append(stats_table)
        else:
            story.append(Paragraph(
                "Numeric columns were found, but no valid numeric "
                "values were available for summary statistics.",
                styles["IFBody"],
            ))
    else:
        story.append(Paragraph(
            "No numeric columns were supplied for statistical analysis.",
            styles["IFBody"],
        ))

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    def add_chart(chart, heading):
        if chart is None:
            return

        try:
            image_bytes = pio.to_image(
                chart,
                format="png",
                width=1100,
                height=600,
                scale=1,
            )

            image_buffer = BytesIO(image_bytes)
            chart_image = PDFImage(
                image_buffer,
                width=165 * mm,
                height=90 * mm,
                kind="proportional",
            )

            story.append(KeepTogether([
                Paragraph(heading, styles["IFSection"]),
                chart_image,
                Spacer(1, 4 * mm),
            ]))

        except Exception as exc:
            story.append(Paragraph(
                f"{safe_text(heading)} could not be embedded in "
                f"the PDF. Check the chart export dependencies. "
                f"Details: {safe_text(str(exc)[:180])}",
                styles["IFBody"],
            ))

    if trend_chart is not None or histogram_chart is not None:
        story.append(Paragraph(
            "3. Visual analysis",
            styles["IFSection"],
        ))

    add_chart(trend_chart, "Trend analysis")
    add_chart(histogram_chart, "Value distribution")

    # -----------------------------------------------------
    # ANOMALY EVIDENCE
    # -----------------------------------------------------

    if anomaly_df is not None:
        if not anomaly_df.empty:
            story.append(Paragraph(
                "4. Detected anomalies",
                styles["IFSection"],
            ))

            story.append(Paragraph(
                f"The anomaly analysis supplied {len(anomaly_df):,} "
                f"rows. The table below shows up to 30 rows for "
                f"readability. Statistical anomalies should be "
                f"investigated in context and are not automatically "
                f"errors or fraudulent records.",
                styles["IFBody"],
            ))

            anomaly_data = [[
                Paragraph(safe_text(column), styles["IFTableHeader"])
                for column in anomaly_df.columns
            ]]

            for _, row in anomaly_df.head(30).iterrows():
                anomaly_data.append([
                    Paragraph(
                        safe_text(str(value)[:100]),
                        styles["IFSmall"],
                    )
                    for value in row.tolist()
                ])

            available_width = A4[0] - (32 * mm)
            column_count_anomaly = max(len(anomaly_df.columns), 1)
            col_width = available_width / column_count_anomaly

            anomaly_table = LongTable(
                anomaly_data,
                colWidths=[col_width] * column_count_anomaly,
                repeatRows=1,
                hAlign="LEFT",
            )
            anomaly_table.setStyle(table_style(font_size=6))
            story.append(anomaly_table)

        else:
            story.append(Paragraph(
                "4. Detected anomalies",
                styles["IFSection"],
            ))
            story.append(Paragraph(
                "The supplied anomaly table is empty. This means no "
                "rows were supplied to this report section; it does "
                "not independently prove that the dataset contains "
                "no unusual observations.",
                styles["IFBody"],
            ))

    # -----------------------------------------------------
    # AI INTERPRETATION AND RECOMMENDATIONS
    # -----------------------------------------------------

    if report_text and str(report_text).strip():
        story.append(PageBreak())

        story.append(Paragraph(
            "5. AI interpretation and recommendations",
            styles["IFSection"],
        ))

        story.append(Paragraph(
            "The following narrative was supplied by the application's "
            "AI report-generation step. Treat recommendations as "
            "investigation guidance and validate important conclusions "
            "against the underlying data.",
            styles["IFBody"],
        ))

        for line in str(report_text).splitlines():
            clean_line = line.strip()

            if not clean_line:
                story.append(Spacer(1, 2 * mm))
                continue

            # Correct regular expressions for Markdown headings.
            heading_match = re.match(
                r"^#{1,6}\s+(.*)$",
                clean_line,
            )

            if heading_match:
                heading_text = inline_markdown(
                    heading_match.group(1)
                )
                story.append(Paragraph(
                    heading_text,
                    styles["IFSubheading"],
                ))
                continue

            # Markdown bullets.
            bullet_match = re.match(
                r"^[-*+]\s+(.*)$",
                clean_line,
            )

            # Markdown numbered lists.
            number_match = re.match(
                r"^(\d+[.)])\s+(.*)$",
                clean_line,
            )

            if bullet_match:
                body = inline_markdown(bullet_match.group(1))
                story.append(Paragraph(
                    f"&bull;&nbsp;&nbsp;{body}",
                    styles["IFBody"],
                ))

            elif number_match:
                number = safe_text(number_match.group(1))
                body = inline_markdown(number_match.group(2))
                story.append(Paragraph(
                    f"<b>{number}</b>&nbsp;&nbsp;{body}",
                    styles["IFBody"],
                ))

            else:
                story.append(Paragraph(
                    inline_markdown(clean_line),
                    styles["IFBody"],
                ))

    # -----------------------------------------------------
    # FOOTER AND PAGE NUMBERS
    # -----------------------------------------------------

    def add_page_number(canvas, document):
        canvas.saveState()

        page_width, _ = A4

        canvas.setStrokeColor(colors.HexColor("#E5EAF1"))
        canvas.setLineWidth(0.5)
        canvas.line(
            16 * mm,
            14 * mm,
            page_width - 16 * mm,
            14 * mm,
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))

        canvas.drawString(
            16 * mm,
            9 * mm,
            "InsightForge | AI Data Detective",
        )

        canvas.drawRightString(
            page_width - 16 * mm,
            9 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    # -----------------------------------------------------
    # BUILD PDF
    # -----------------------------------------------------

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
    
# =========================================================
# HEADER
# =========================================================

# =========================================================
# INSIGHTFORGE — JULIUS-INSPIRED HOMEPAGE
# =========================================================

# Sidebar navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 4px 28px;">
            <div style="font-size: 25px; font-weight: 800;
                        letter-spacing: -1px; color: #315efb;">
                InsightForge
            </div>
            <div style="font-size: 12px; color: #788397;
                        margin-top: 5px;">
                AI DATA WORKSPACE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Overview",
            "Detect",
            "Investigate",
            "Ask Your Data",
            "Reports",
        ],
        label_visibility="collapsed",
        key="if_navigation",
    )

    st.divider()

    st.markdown("#### Your workflow")

    st.markdown(
        """
        <div style="font-size: 13px; line-height: 2.2;
                    color: #69758a;">
            <div>01 &nbsp; Upload your data</div>
            <div>02 &nbsp; Detect patterns</div>
            <div>03 &nbsp; Investigate findings</div>
            <div>04 &nbsp; Generate a report</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Powered by Python, Plotly and Gemma AI")


# Main homepage
st.markdown(
    """
    <div style="display:flex; justify-content:space-between;
                align-items:center; padding:8px 0 24px;
                border-bottom:1px solid #e7eaf0;">
        <div style="font-size:14px; color:#69758a;">
            InsightForge &nbsp; / &nbsp; Data workspace
        </div>
        <div style="font-size:12px; color:#315efb;
                    background:#edf2ff; padding:7px 12px;
                    border-radius:20px;">
            AI-powered analytics
        </div>
    </div>

    <div style="text-align:center; padding:100px 12px 20px;">
        <div style="font-size:12px; font-weight:700;
                    letter-spacing:2px; color:#315efb;
                    margin-bottom:20px;">
            YOUR AI DATA DETECTIVE
        </div>
        <div style="font-family:Manrope, sans-serif;
                    font-size:clamp(32px, 4vw, 46px);
                    font-weight:800; letter-spacing:-1.8px;
                    line-height:1.2; color:#172033;">
            What can I help you<br>discover today?
        </div>
        <div style="max-width:560px; margin:18px auto 0;
                    color:#788397; font-size:16px;
                    line-height:1.8;">
            Upload a dataset to uncover anomalies, understand
            trends, investigate unusual results, and turn data
            into actionable insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dataset upload panel
left, center, right = st.columns([1, 5, 1])

with center:
    st.markdown(
        """
        <div style="background:white; border:1px solid #e4e8ef;
                    border-radius:18px 18px 0 0;
                    padding:20px 22px 4px;">
            <div style="font-size:16px; font-weight:600;
                        color:#172033;">
                Start with your data
            </div>
            <div style="font-size:13px; color:#788397;
                        margin-top:5px;">
                Upload a CSV or Excel file to begin your analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="insightforge_home_upload",
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="background:white; border:1px solid #e4e8ef;
                    border-top:0; border-radius:0 0 18px 18px;
                    padding:4px 20px 18px;
                    color:#8a94a6; font-size:12px;">
            Supported formats: CSV and XLSX
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div style="text-align:center; color:#69758a;
                    font-size:12px; margin:12px 0 18px;">
            OR EXPLORE A WORKFLOW
        </div>
        """,
        unsafe_allow_html=True,
    )

    action1, action2 = st.columns(2)

    with action1:
        if st.button(
            "Detect anomalies",
            use_container_width=True,
            key="if_detect_home",
        ):
            st.session_state["if_navigation"] = "Detect"
            st.rerun()

    with action2:
        if st.button(
            "Investigate data",
            use_container_width=True,
            key="if_investigate_home",
        ):
            st.session_state["if_navigation"] = "Investigate"
            st.rerun()

    action3, action4 = st.columns(2)

    with action3:
        if st.button(
            "Ask your data",
            use_container_width=True,
            key="if_ask_home",
        ):
            st.session_state["if_navigation"] = "Ask Your Data"
            st.rerun()

    with action4:
        if st.button(
            "Generate report",
            use_container_width=True,
            key="if_report_home",
        ):
            st.session_state["if_navigation"] = "Reports"
            st.rerun()


if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center; margin-top:42px;
                    color:#9aa3b2; font-size:12px;">
            Detect &nbsp; · &nbsp; Investigate &nbsp; · &nbsp; Explain
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    try:
        # Keep optional PDF sections safe when a dataset lacks a chart/anomaly.
        trend_chart_for_pdf = None
        histogram_chart_for_pdf = None
        anomaly_df_for_pdf = pd.DataFrame()

        # =================================================
        # READ DATA
        # =================================================

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            f"✅ Successfully loaded **{uploaded_file.name}**"
        )

        # =================================================
        # DATASET OVERVIEW
        # =================================================

        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        col4.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

        # =================================================
        # DATA PREVIEW
        # =================================================

        st.header("📋 Data Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        # =================================================
        # DATA QUALITY
        # =================================================

        st.header("⚠️ Data Quality")

        missing_values = df.isnull().sum()

        missing_values = missing_values[
            missing_values > 0
        ]

        if len(missing_values) == 0:

            st.success(
                "✅ No missing values detected."
            )

        else:

            st.warning(
                f"⚠️ {int(missing_values.sum())} "
                "missing values detected."
            )

            missing_table = pd.DataFrame({
                "Column": missing_values.index,
                "Missing Values": missing_values.values,
                "Missing %": (
                    missing_values.values
                    / len(df)
                    * 100
                ).round(2)
            })

            st.dataframe(
                missing_table,
                use_container_width=True
            )

        # -------------------------------------------------
        # DUPLICATES
        # -------------------------------------------------

        duplicate_rows = int(
            df.duplicated().sum()
        )

        if duplicate_rows == 0:

            st.success(
                "✅ No duplicate rows detected."
            )

        else:

            st.warning(
                f"⚠️ {duplicate_rows} duplicate rows detected."
            )

        # =================================================
        # COLUMN INFORMATION
        # =================================================

        st.header("🔍 Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": [
                int(df[column].isnull().sum())
                for column in df.columns
            ],
            "Unique Values": [
                int(df[column].nunique())
                for column in df.columns
            ]
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        # =================================================
        # NUMERIC COLUMNS
        # =================================================

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        # =================================================
        # BASIC STATISTICS
        # =================================================

        st.header("📈 Statistical Summary")

        if numeric_columns:

            statistics = df[
                numeric_columns
            ].describe().T

            statistics["Range"] = (
                statistics["max"]
                - statistics["min"]
            )

            st.dataframe(
                statistics,
                use_container_width=True
            )

        else:

            st.info(
                "No numeric columns available."
            )

        # =================================================
        # QUICK INSIGHTS
        # =================================================

        st.header("💡 Quick Insights")

        if numeric_columns:

            for column in numeric_columns:

                values = df[column].dropna()

                if len(values) == 0:
                    continue

                average = values.mean()
                minimum = values.min()
                maximum = values.max()
                median = values.median()

                st.write(
                    f"**{column}** — "
                    f"Average: `{average:,.2f}` | "
                    f"Median: `{median:,.2f}` | "
                    f"Minimum: `{minimum:,.2f}` | "
                    f"Maximum: `{maximum:,.2f}`"
                )

        # =================================================
        # AUTOMATIC TREND DETECTION
        # =================================================

        st.header("📈 Automatic Trend Detection")

        date_columns = []

        for column in df.columns:

            # Only attempt automatic conversion
            # for non-numeric columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                continue

            try:

                converted_dates = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = (
                    converted_dates.notna().mean()
                )

                if valid_ratio >= 0.7:

                    date_columns.append(column)

            except Exception:
                pass

        trend_df = None
        trend_date_column = None
        trend_value_column = None

        if date_columns and numeric_columns:

            trend_col1, trend_col2 = st.columns(2)

            with trend_col1:

                trend_date_column = st.selectbox(
                    "Date column",
                    date_columns,
                    key="trend_date"
                )

            with trend_col2:

                trend_value_column = st.selectbox(
                    "Value column",
                    numeric_columns,
                    key="trend_value"
                )

            trend_df = df.copy()

            trend_df[
                trend_date_column
            ] = pd.to_datetime(
                trend_df[trend_date_column],
                errors="coerce"
            )

            trend_df = trend_df.dropna(
                subset=[
                    trend_date_column,
                    trend_value_column
                ]
            )

            trend_df = trend_df.sort_values(
                trend_date_column
            )

            if len(trend_df) >= 2:

                first_value = trend_df[
                    trend_value_column
                ].iloc[0]

                last_value = trend_df[
                    trend_value_column
                ].iloc[-1]

                if first_value != 0:

                    percentage_change = (
                        (
                            last_value
                            - first_value
                        )
                        / abs(first_value)
                    ) * 100

                    if percentage_change > 0:

                        st.success(
                            f"📈 **{trend_value_column}** "
                            f"increased by "
                            f"**{percentage_change:.1f}%** "
                            f"from the first to the last "
                            f"observed value."
                        )

                    elif percentage_change < 0:

                        st.warning(
                            f"📉 **{trend_value_column}** "
                            f"decreased by "
                            f"**{abs(percentage_change):.1f}%** "
                            f"from the first to the last "
                            f"observed value."
                        )

                    else:

                        st.info(
                            f"➡️ **{trend_value_column}** "
                            "was unchanged between the "
                            "first and last observations."
                        )

                trend_chart = px.line(
                    trend_df,
                    x=trend_date_column,
                    y=trend_value_column,
                    markers=True,
                    title=(
                        f"{trend_value_column} "
                        "Over Time"
                    )
                )

                st.plotly_chart(
                    trend_chart,
                    use_container_width=True
                )
                trend_chart_for_pdf = trend_chart

        else:

            st.info(
                "No suitable date and numeric columns "
                "were found for trend detection."
            )

        # =================================================
        # SMART ANOMALY DETECTION
        # =================================================

        st.header("🚨 Smart Anomaly Detection")

        anomaly_results = []

        anomaly_details = {}

        if numeric_columns:

            for column in numeric_columns:

                values = df[column].dropna()

                if len(values) < 3:
                    continue

                # -----------------------------------------
                # IQR DETECTION
                # -----------------------------------------

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)

                iqr = q3 - q1

                lower_bound = (
                    q1 - 1.5 * iqr
                )

                upper_bound = (
                    q3 + 1.5 * iqr
                )

                iqr_mask = (
                    (df[column] < lower_bound)
                    |
                    (df[column] > upper_bound)
                )

                # -----------------------------------------
                # Z-SCORE DETECTION
                # -----------------------------------------

                mean_value = values.mean()
                std_value = values.std()

                if std_value != 0:

                    z_scores = (
                        (
                            df[column]
                            - mean_value
                        )
                        / std_value
                    )

                    z_mask = (
                        z_scores.abs() > 2
                    )

                else:

                    z_scores = pd.Series(
                        0,
                        index=df.index
                    )

                    z_mask = pd.Series(
                        False,
                        index=df.index
                    )

                # -----------------------------------------
                # COMBINE METHODS
                # -----------------------------------------

                combined_mask = (
                    iqr_mask | z_mask
                )

                anomaly_rows = df[
                    combined_mask
                ].copy()

                if len(anomaly_rows) > 0:

                    anomaly_details[column] = (
                        anomaly_rows
                    )

                    for index in anomaly_rows.index:

                        value = df.loc[
                            index,
                            column
                        ]

                        z_score = z_scores.loc[
                            index
                        ]

                        methods = []

                        if iqr_mask.loc[index]:
                            methods.append("IQR")

                        if z_mask.loc[index]:
                            methods.append(
                                "Z-score"
                            )

                        anomaly_results.append({
                            "Column": column,
                            "Row": int(index),
                            "Value": value,
                            "Z-Score": round(
                                float(z_score),
                                2
                            ),
                            "Detection": " + ".join(
                                methods
                            )
                        })

            # ---------------------------------------------
            # DISPLAY ANOMALIES
            # ---------------------------------------------

            if anomaly_results:

                anomaly_df = pd.DataFrame(
                    anomaly_results
                )
                anomaly_df_for_pdf = anomaly_df.copy()

                st.warning(
                    f"🚨 **{len(anomaly_df)} "
                    "potential anomalous values "
                    "detected.**"
                )

                st.dataframe(
                    anomaly_df,
                    use_container_width=True
                )

            else:

                st.success(
                    "✅ No major statistical anomalies "
                    "detected."
                )

        else:

            st.info(
                "No numeric columns available "
                "for anomaly detection."
            )

        # =================================================
        # DATA VISUALIZATION
        # =================================================

        st.header("📊 Explore Your Data")

        if numeric_columns:

            chart_column = st.selectbox(
                "Choose a numeric column",
                numeric_columns,
                key="distribution_column"
            )

            histogram = px.histogram(
                df,
                x=chart_column,
                nbins=30,
                title=(
                    f"Distribution of "
                    f"{chart_column}"
                )
            )

            st.plotly_chart(
                histogram,
                use_container_width=True
            )
            histogram_chart_for_pdf = histogram

        # =================================================
        # INVESTIGATE ANOMALY
        # =================================================

        st.header("🔎 Investigate This")

        if anomaly_results:

            investigation_options = [
                (
                    f"{item['Column']} | "
                    f"Row {item['Row']} | "
                    f"{item['Value']}"
                )
                for item in anomaly_results
            ]

            selected_anomaly = st.selectbox(
                "Select an anomaly to investigate",
                investigation_options,
                key="selected_anomaly"
            )

            selected_position = (
                investigation_options.index(
                    selected_anomaly
                )
            )

            selected_anomaly_data = (
                anomaly_results[
                    selected_position
                ]
            )

            selected_column = (
                selected_anomaly_data[
                    "Column"
                ]
            )

            selected_row_index = (
                selected_anomaly_data[
                    "Row"
                ]
            )

            selected_value = (
                selected_anomaly_data[
                    "Value"
                ]
            )

            if st.button(
                "🕵️ Investigate Selected Anomaly"
            ):

                selected_row = df.loc[
                    selected_row_index
                ]

                column_values = df[
                    selected_column
                ].dropna()

                mean_value = (
                    column_values.mean()
                )

                median_value = (
                    column_values.median()
                )

                investigation_prompt = f"""
You are InsightForge, an AI Data Detective.

Investigate a statistically unusual data point.

IMPORTANT:
Only make factual claims using the evidence
provided below.

Do not invent information.

Clearly separate:
- Observed facts
- Possible explanations
- Recommended next checks

DATASET:
Rows: {len(df)}
Columns: {len(df.columns)}

INVESTIGATED COLUMN:
{selected_column}

ANOMALOUS VALUE:
{selected_value}

COLUMN MEAN:
{mean_value}

COLUMN MEDIAN:
{median_value}

Z-SCORE:
{selected_anomaly_data["Z-Score"]}

DETECTION METHOD:
{selected_anomaly_data["Detection"]}

FULL RECORD:
{selected_row.to_string()}

Provide:

1. Why this value was flagged.
2. What the surrounding record tells us.
3. What is statistically unusual.
4. Possible explanations.
5. Three specific things the user should investigate next.

Do not claim that an anomaly is an error unless
the data proves it.
"""

                with st.spinner(
                    "🤖 Gemma is investigating the anomaly..."
                ):

                    response = (
                        client.models.generate_content(
                            model=MODEL_NAME,
                            contents=investigation_prompt
                        )
                    )

                st.subheader(
                    "🧠 AI Anomaly Investigation"
                )

                st.write(
                    response.text
                )

        elif numeric_columns:

            st.info(
                "No anomalies are currently available "
                "for investigation."
            )

        # =================================================
        # INVESTIGATE HIGHEST VALUE
        # =================================================

        st.subheader(
            "💰 Investigate Highest Value"
        )

        if numeric_columns:

            highest_column = st.selectbox(
                "Choose a numeric column",
                numeric_columns,
                key="highest_value_column"
            )

            if st.button(
                "🔎 Investigate Highest Value",
                key="highest_value_button"
            ):

                highest_index = df[
                    highest_column
                ].idxmax()

                highest_row = df.loc[
                    highest_index
                ]

                highest_value = highest_row[
                    highest_column
                ]

                st.info(
                    f"Highest **{highest_column}** "
                    f"value: **{highest_value}**"
                )

                st.dataframe(
                    highest_row.to_frame("Value"),
                    use_container_width=True
                )

                highest_prompt = f"""
You are InsightForge, an AI Data Detective.

Investigate the record containing the highest
value for the selected column.

COLUMN:
{highest_column}

HIGHEST VALUE:
{highest_value}

FULL RECORD:
{highest_row.to_string()}

COLUMN STATISTICS:

Mean:
{df[highest_column].mean()}

Median:
{df[highest_column].median()}

Minimum:
{df[highest_column].min()}

Maximum:
{df[highest_column].max()}

Explain:

1. What makes this record notable.
2. How it compares with the average.
3. Whether it was also detected as a statistical anomaly.
4. Possible explanations.
5. What should be investigated next.

Only use evidence from the supplied data.
"""

                with st.spinner(
                    "🤖 Gemma is investigating..."
                ):

                    highest_response = (
                        client.models.generate_content(
                            model=MODEL_NAME,
                            contents=highest_prompt
                        )
                    )

                st.subheader(
                    "🧠 AI Investigation"
                )

                st.write(
                    highest_response.text
                )

        # =================================================
        # ASK YOUR DATA — CONVERSATIONAL ANALYSIS
        # =================================================

        st.header("💬 Ask Your Data")
        st.write(
            "Ask questions in plain English. Python computes the result first, "
            "then Gemma 4 explains the verified evidence."
        )
        st.caption(
            "Try: Which region has the highest total sales? · What is the average Sales? · "
            "Are there missing values? · How did Sales change over time?"
        )

        try:
            dataset_signature = (
                f"{uploaded_file.name}:{len(df)}:{len(df.columns)}:"
                f"{int(pd.util.hash_pandas_object(df, index=True).sum())}"
            )
        except Exception:
            dataset_signature = f"{uploaded_file.name}:{len(df)}:{tuple(map(str, df.columns))}"

        if st.session_state.get("ask_data_dataset_signature") != dataset_signature:
            st.session_state["ask_data_dataset_signature"] = dataset_signature
            st.session_state["ask_data_messages"] = []

        for message_index, chat_message in enumerate(st.session_state.get("ask_data_messages", [])):
            with st.chat_message(chat_message["role"]):
                st.markdown(chat_message["content"])
                if chat_message["role"] == "assistant" and chat_message.get("evidence") is not None:
                    with st.expander("🔬 View verified Python evidence"):
                        evidence = chat_message["evidence"]
                        if isinstance(evidence, dict) and isinstance(evidence.get("results"), list):
                            st.dataframe(pd.DataFrame(evidence["results"]), use_container_width=True)
                        elif isinstance(evidence, dict) and isinstance(evidence.get("rows"), list):
                            st.dataframe(pd.DataFrame(evidence["rows"]), use_container_width=True)
                        else:
                            st.json(evidence)

        suggested_questions = st.columns(3)
        numeric_for_questions = list(df.select_dtypes(include=np.number).columns)
        categorical_for_questions = [
            c for c in df.columns
            if c not in numeric_for_questions and df[c].nunique(dropna=True) <= min(50, max(10, len(df) // 2))
        ]
        if numeric_for_questions and categorical_for_questions:
            example_questions = [
                f"Which {categorical_for_questions[0]} has the highest total {numeric_for_questions[0]}?",
                f"What is the average {numeric_for_questions[0]}?",
                "Are there missing values?",
            ]
        elif numeric_for_questions:
            example_questions = [
                "Summarize this dataset",
                f"What is the average {numeric_for_questions[0]}?",
                "Are there missing values?",
            ]
        else:
            first_column = str(df.columns[0]) if len(df.columns) else "data"
            example_questions = [
                "Summarize this dataset",
                f"What are the most common values in {first_column}?",
                "Are there missing values?",
            ]
        for idx, example in enumerate(example_questions):
            if suggested_questions[idx].button(example, key=f"ask_data_example_{idx}"):
                st.session_state["ask_data_pending_question"] = example

        # Use an explicit form for typed questions. This is more reliable than
        # st.chat_input in nested app layouts and makes submission visible to users.
        typed_question = ""
        submitted_question = False
        with st.form(key="ask_data_question_form", clear_on_submit=True):
            typed_question = st.text_input(
                "Type your own question",
                placeholder="e.g. Which region has the highest total Sales?",
                key="ask_data_text_question",
            )
            submitted_question = st.form_submit_button("Ask InsightForge", type="primary")

        question = typed_question if submitted_question else None
        if st.session_state.get("ask_data_pending_question"):
            question = st.session_state.pop("ask_data_pending_question")

        if question:
            question = question.strip()
            if question:
                st.session_state["ask_data_messages"].append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Mapping your question to a safe analysis…"):
                            analysis_plan = _create_analysis_plan(question, df)

                        if not isinstance(analysis_plan, dict) or not analysis_plan.get("operation"):
                            raise ValueError(
                                "I couldn't determine the requested analysis. "
                                "Please include the column name, such as "
                                "'Find the minimum Sales value and its Date'."
                            )
                        with st.spinner("Python is calculating verified results…"):
                            computed_evidence = _execute_analysis_plan(df, analysis_plan)
                        with st.spinner("Gemma is explaining the computed evidence…"):
                            answer_text = _answer_data_question(question, computed_evidence)
                        st.markdown(answer_text)
                        with st.expander("🔬 View verified Python evidence"):
                            if isinstance(computed_evidence, dict) and isinstance(computed_evidence.get("results"), list):
                                st.dataframe(pd.DataFrame(computed_evidence["results"]), use_container_width=True)
                            elif isinstance(computed_evidence, dict) and isinstance(computed_evidence.get("rows"), list):
                                st.dataframe(pd.DataFrame(computed_evidence["rows"]), use_container_width=True)
                            else:
                                st.json(computed_evidence)
                        st.session_state["ask_data_messages"].append({
                            "role": "assistant", "content": answer_text, "evidence": computed_evidence
                        })
                    except Exception as ask_error:
                        safe_error = (
                            "I couldn't complete that analysis. Try asking about summary, column statistics, "
                            "highest/lowest totals by group, trends, missing values, duplicates, or correlation.\n\n"
                            f"Details: {str(ask_error)}"
                        )
                        st.error(safe_error)
                        st.session_state["ask_data_messages"].append({
                            "role": "assistant", "content": safe_error, "evidence": None
                        })

        if st.session_state.get("ask_data_messages"):
            if st.button("🗑️ Clear Ask Your Data conversation", key="clear_ask_data_chat"):
                st.session_state["ask_data_messages"] = []
                st.rerun()

        # =================================================
        # BUILD VERIFIED DATA SUMMARY FOR GEMMA
        # =================================================

        st.header("🤖 AI Data Detective")

        st.write(
            "Gemma receives verified statistical findings "
            "from Python before generating its explanation."
        )

        if st.button(
            "🧠 Generate Full AI Insight Report"
        ):

            # ---------------------------------------------
            # Build verified findings
            # ---------------------------------------------

            verified_findings = []

            verified_findings.append(
                f"Dataset contains {len(df)} rows "
                f"and {len(df.columns)} columns."
            )

            verified_findings.append(
                f"Total missing values: "
                f"{int(df.isnull().sum().sum())}."
            )

            verified_findings.append(
                f"Duplicate rows: "
                f"{int(df.duplicated().sum())}."
            )

            # Numeric statistics

            for column in numeric_columns:

                values = df[column].dropna()

                if len(values) == 0:
                    continue

                verified_findings.append(
                    f"{column}: "
                    f"mean={values.mean():.2f}, "
                    f"median={values.median():.2f}, "
                    f"min={values.min():.2f}, "
                    f"max={values.max():.2f}."
                )

            # Anomalies

            if anomaly_results:

                verified_findings.append(
                    f"Detected "
                    f"{len(anomaly_results)} "
                    "potential anomalous values."
                )

                for anomaly in anomaly_results[
                    :20
                ]:

                    verified_findings.append(
                        f"Anomaly: "
                        f"{anomaly['Column']} = "
                        f"{anomaly['Value']} "
                        f"(Z-score "
                        f"{anomaly['Z-Score']}, "
                        f"method "
                        f"{anomaly['Detection']})."
                    )

            else:

                verified_findings.append(
                    "No major statistical anomalies "
                    "were detected."
                )

            # Trend

            if (
                trend_df is not None
                and trend_date_column is not None
                and trend_value_column is not None
                and len(trend_df) >= 2
            ):

                first_value = trend_df[
                    trend_value_column
                ].iloc[0]

                last_value = trend_df[
                    trend_value_column
                ].iloc[-1]

                if first_value != 0:

                    change = (
                        (
                            last_value
                            - first_value
                        )
                        / abs(first_value)
                    ) * 100

                    verified_findings.append(
                        f"Trend: {trend_value_column} "
                        f"changed by {change:.2f}% "
                        f"between the first and last "
                        f"observed dates."
                    )

            findings_text = "\n".join(
                verified_findings
            )

            # ---------------------------------------------
            # GEMMA PROMPT
            # ---------------------------------------------

            report_prompt = f"""
You are InsightForge, an AI Data Detective.

Generate a concise data investigation report (roughly 500–700 words maximum). Avoid repeating the same finding in multiple sections.

IMPORTANT RULES:

1. Use only the verified findings below.
2. Do not invent statistics.
3. Do not claim causation without evidence.
4. Clearly distinguish facts from possibilities.
5. If evidence is insufficient, say so.
6. Give practical next steps.

VERIFIED PYTHON FINDINGS:

{findings_text}

Generate the report using these sections:

## 🔎 Executive Summary

Summarize the most important findings.

## 📈 Trends

Explain the detected trend information.

## 🚨 Anomalies

Explain unusual values and their statistical evidence.

## ⚠️ Data Quality

Discuss missing values and duplicate rows.

## 💡 Business Insights

Provide useful interpretations that are
supported by the evidence.

## 🕵️ What Should Be Investigated Next?

Give three concrete investigation steps.

Remember:
Facts must come from the supplied findings.
Possible explanations must be clearly labeled
as possibilities.
"""

            with st.spinner(
                "🤖 Gemma is generating your report..."
            ):

                report_response = (
                    client.models.generate_content(
                        model=MODEL_NAME,
                        contents=report_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=900
                        )
                    )
                )

            report_text = report_response.text

            st.markdown(
                report_text
            )

            # ---------------------------------------------
            # DOWNLOAD REPORT: PDF WITH VERIFIED EVIDENCE
            # ---------------------------------------------

            st.success(
                "Your AI report is ready. Download the evidence-rich PDF below."
            )

            try:
                pdf_report = build_pdf_report(
                    dataset_name=uploaded_file.name,
                    df=df,
                    numeric_columns=numeric_columns,
                    report_text=report_text,
                    trend_chart=trend_chart_for_pdf,
                    histogram_chart=histogram_chart_for_pdf,
                    anomaly_df=anomaly_df_for_pdf,
                )

                st.download_button(
                    label="📄 Download Evidence Report (PDF)",
                    data=pdf_report,
                    file_name="insightforge_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                )

            except Exception as pdf_error:
                st.error(f"Could not generate the PDF report: {pdf_error}")
                st.info(
                    "You can still download the AI interpretation as Markdown."
                )

            report_download = (
                f"""# InsightForge AI Data Report

Dataset: {uploaded_file.name}

{report_text}
"""
            )

            st.download_button(
                label="⬇️ Download AI Report (Markdown)",
                data=report_download,
                file_name="insightforge_report.md",
                mime="text/markdown",
                key="download_markdown_report",
            )

        # =================================================
        # RAW DATA DOWNLOAD
        # =================================================

        st.header("⬇️ Export")

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Dataset as CSV",
            data=csv_data,
            file_name="insightforge_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Something went wrong while processing "
            "your dataset."
        )

        st.exception(e)

# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "👆 Upload a CSV or Excel file above "
        "to start investigating your data."
    )

    st.write("### 🚀 How InsightForge Works")

    step1, step2, step3, step4 = st.columns(4)

    step1.markdown(
        "**1️⃣ Upload**\n\n"
        "Upload your CSV or Excel dataset."
    )

    step2.markdown(
        "**2️⃣ Detect**\n\n"
        "Python automatically finds patterns, "
        "trends and anomalies."
    )

    step3.markdown(
        "**3️⃣ Investigate**\n\n"
        "Select an interesting finding "
        "and investigate it with AI."
    )

    step4.markdown(
        "**4️⃣ Understand**\n\n"
        "Gemma explains the evidence and "
        "suggests what to investigate next."
    )
