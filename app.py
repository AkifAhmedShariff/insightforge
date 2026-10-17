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
    page_title="InsightForge | AI Data Analyst",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
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
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The AI returned an empty analysis plan.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try parsing the complete response first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON object from surrounding explanatory text.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The AI could not produce a valid analysis plan. Please rephrase the question."
    )



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
    else:
        raise ValueError("I couldn't map that question to a supported analysis. Try asking about summary, statistics, groups, trends, missing values, duplicates, or correlation.")

    return _json_safe(result)



def _create_analysis_plan(question, df):
    """Use fast local rules for common questions; call Gemma only as a fallback."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a non-empty question.")

    if df is None or not hasattr(df, "columns"):
        raise ValueError("A valid dataset is required for analysis.")

    if df.empty:
        raise ValueError("The dataset contains no rows to analyze.")

    q = question.strip().lower()
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

    # Fall back to Gemma for less common or ambiguous natural-language questions.
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

Use only exact column names from the dataset context. If the user asks which category/region/product has the highest or lowest value, use aggregate with the appropriate value column and sum/mean as the wording implies. If they ask for most frequent categories, use value_counts. For a general question, choose dataset_summary. Return ONLY a JSON object. Dataset context: {json.dumps(context, ensure_ascii=False, default=str)}
User question: {question}
JSON plan:"""
    response = client.models.generate_content(
        model=MODEL_NAME, contents=planner_prompt,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=180)
    )
    plan = _extract_json_object(response.text)
    return plan


def _local_evidence_answer(question, evidence):
    """Fast, deterministic answer for common questions using Python-computed evidence."""
    if not isinstance(evidence, dict):
        return "Python completed the analysis, but the result was not in a format I can summarize."

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
# CUSTOM STYLE
# =========================================================


# =========================================================
# CUSTOM STYLE — EDITORIAL MINIMAL
# =========================================================

st.markdown(
    """
    <style>

    /* Readability fixes for the light theme */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        color: #17212f;
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    [data-testid="stCaptionContainer"] {
        color: #364152;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17212f !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff;
        color: #17212f;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: #364152 !important;
    }

    [data-testid="stAlert"] {
        color: #17212f;
    }

    [data-testid="stAlert"] p {
        color: inherit;
    }

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17212f;
        --muted: #687588;
        --line: #e5eaf1;
        --blue: #315efb;
        --paper: #f7f9fc;
        --surface: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.5px;
    }

    .main .block-container {
        max-width: 1440px;
        padding: 2.2rem 3rem 4rem;
    }

    .main-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(34px, 4vw, 48px);
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1.15;
        color: var(--ink);
    }

    .subtitle {
        font-size: 16px;
        line-height: 1.7;
        color: var(--muted);
        max-width: 760px;
        margin-top: 10px;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(25, 42, 70, 0.025);
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

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.7px;
    }

    h2 {
        font-size: 25px !important;
        margin-top: 1.5rem !important;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--blue);
        border: 1px solid var(--blue);
        color: white;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--blue);
        color: var(--blue);
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 12px;
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.2rem 1rem 3rem;
        }

        .main-title {
            letter-spacing: -1px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PDF REPORT GENERATOR
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
    """Build an evidence-based PDF report in memory."""
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"InsightForge Report - {dataset_name}",
        author="InsightForge",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#17212F"),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#687588"),
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#315EFB"),
        spaceBefore=16,
        spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=7,
    ))

    story = []

    story.append(Paragraph("InsightForge", styles["ReportTitle"]))
    story.append(Paragraph(
        "AI DATA DETECTIVE  /  DATA INVESTIGATION REPORT",
        styles["ReportSubtitle"],
    ))
    story.append(Paragraph(
        f"<b>Dataset:</b> {dataset_name}<br/>"
        f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}<br/>"
        f"<b>Records:</b> {len(df):,} &nbsp;&nbsp; "
        f"<b>Columns:</b> {len(df.columns):,}",
        styles["ReportBody"],
    ))

    story.append(Paragraph("Dataset at a glance", styles["SectionHeading"]))

    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    numeric_count = len(numeric_columns)

    kpi_data = [
        ["ROWS", "COLUMNS", "MISSING VALUES", "DUPLICATE ROWS"],
        [
            f"{len(df):,}",
            f"{len(df.columns):,}",
            f"{missing_total:,}",
            f"{duplicate_total:,}",
        ],
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[40 * mm, 40 * mm, 40 * mm, 40 * mm],
    )
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#17212F")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5EAF1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5EAF1")),
    ]))
    story.append(kpi_table)

    story.append(Paragraph(
        "Data quality details",
        styles["SectionHeading"],
    ))

    quality_data = [["Column", "Data type", "Missing", "Missing %", "Unique"]]
    for column in df.columns:
        missing = int(df[column].isna().sum())
        missing_pct = (missing / len(df) * 100) if len(df) else 0
        quality_data.append([
            str(column),
            str(df[column].dtype),
            str(missing),
            f"{missing_pct:.1f}%",
            str(df[column].nunique()),
        ])

    quality_table = Table(
        quality_data,
        repeatRows=1,
        colWidths=[43 * mm, 35 * mm, 22 * mm, 25 * mm, 25 * mm],
    )
    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17212F")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE4EE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F7F9FC"),
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(quality_table)

    if numeric_columns:
        story.append(Paragraph(
            "Statistical evidence",
            styles["SectionHeading"],
        ))

        stats_data = [[
            "Column", "Count", "Mean", "Median", "Minimum", "Maximum"
        ]]
        for column in numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if values.empty:
                continue
            stats_data.append([
                str(column),
                f"{len(values):,}",
                f"{values.mean():,.2f}",
                f"{values.median():,.2f}",
                f"{values.min():,.2f}",
                f"{values.max():,.2f}",
            ])

        if len(stats_data) > 1:
            stats_table = Table(
                stats_data,
                repeatRows=1,
                colWidths=[32 * mm, 19 * mm, 29 * mm, 29 * mm, 29 * mm, 29 * mm],
            )
            stats_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE4EE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(stats_table)

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
            )
            # Keep each chart title with its chart to avoid orphan headings.
            story.append(KeepTogether([
                Paragraph(heading, styles["SectionHeading"]),
                chart_image,
                Spacer(1, 5 * mm),
            ]))
        except Exception as exc:
            story.append(Paragraph(
                f"{heading}: chart could not be embedded ({str(exc)[:180]}).",
                styles["ReportBody"],
            ))

    add_chart(trend_chart, "Trend analysis")
    add_chart(histogram_chart, "Value distribution")

    if anomaly_df is not None and not anomaly_df.empty:
        story.append(Paragraph(
            "Detected anomalies — statistical evidence",
            styles["SectionHeading"],
        ))

        anomaly_data = [[
            str(column) for column in anomaly_df.columns
        ]]
        for _, row in anomaly_df.head(30).iterrows():
            anomaly_data.append([
                str(value)[:60] for value in row.tolist()
            ])

        anomaly_table = Table(anomaly_data, repeatRows=1)
        anomaly_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE4EE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(anomaly_table)

    if report_text:
        # Start the narrative on a fresh page so the section title is not
        # stranded at the bottom of the anomaly/chart page.
        story.append(PageBreak())
        story.append(Paragraph(
            "AI interpretation and recommendations",
            styles["SectionHeading"],
        ))

        def markdown_inline_to_reportlab(value):
            """Escape text safely, then render common Markdown emphasis."""
            import html
            safe = html.escape(value, quote=False)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            safe = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe)
            safe = re.sub(r"`([^`]+?)`", r'<font name="Courier">\1</font>', safe)
            return safe

        for line in report_text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                story.append(Spacer(1, 2 * mm))
                continue

            # Render Markdown headings without exposing the ## markers.
            heading_match = re.match(r"^#{1,6}\\s+(.*)$", clean_line)
            if heading_match:
                heading_text = markdown_inline_to_reportlab(heading_match.group(1))
                story.append(Paragraph(heading_text, styles["Heading3"]))
                continue

            # Render bullets and numbered list items as proper PDF list text.
            bullet_match = re.match(r"^[-*+]\\s+(.*)$", clean_line)
            number_match = re.match(r"^(\\d+[.)])\\s+(.*)$", clean_line)
            if bullet_match:
                body = markdown_inline_to_reportlab(bullet_match.group(1))
                story.append(Paragraph(f"&bull;&nbsp;&nbsp;{body}", styles["ReportBody"]))
            elif number_match:
                number = html.escape(number_match.group(1))
                body = markdown_inline_to_reportlab(number_match.group(2))
                story.append(Paragraph(f"<b>{number}</b>&nbsp;&nbsp;{body}", styles["ReportBody"]))
            else:
                story.append(Paragraph(
                    markdown_inline_to_reportlab(clean_line),
                    styles["ReportBody"],
                ))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#E5EAF1"))
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687588"))
        canvas.drawString(18 * mm, 9 * mm, "InsightForge · AI Data Detective")
        canvas.drawRightString(
            192 * mm, 9 * mm, f"Page {document.page}"
        )
        canvas.restoreState()

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

st.markdown(
    '<div class="main-title">🔎 InsightForge</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI Data Detective — Discover what your data is trying to tell you.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.write(
    "Upload a CSV or Excel dataset and InsightForge will "
    "automatically detect patterns, trends, anomalies, "
    "data-quality issues and actionable insights."
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ InsightForge")

    st.write(
        "Your AI-powered data investigation assistant."
    )

    st.divider()

    st.write("### 🔍 Detection Engine")

    st.write("✓ Data profiling")
    st.write("✓ Data quality")
    st.write("✓ Trend detection")
    st.write("✓ IQR anomaly detection")
    st.write("✓ Z-score anomaly detection")
    st.write("✓ AI investigation")
    st.write("✓ Automated insights")

    st.divider()

    st.caption(
        "Powered by Python, Pandas, Plotly and Gemma 4."
    )

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"]
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
                        with st.spinner("Gemma is mapping your question to a safe analysis…"):
                            analysis_plan = _create_analysis_plan(question, df)
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
