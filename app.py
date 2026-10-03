import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from google import genai
import os
from io import StringIO

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
# CUSTOM STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
    }

    .subtitle {
        font-size: 20px;
        color: #666;
    }

    .insight-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

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

Generate a concise but useful data investigation report.

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
                        contents=report_prompt
                    )
                )

            report_text = report_response.text

            st.markdown(
                report_text
            )

            # ---------------------------------------------
            # DOWNLOAD REPORT
            # ---------------------------------------------

            report_download = (
                f"""# InsightForge AI Data Report

Dataset: {uploaded_file.name}

{report_text}
"""
            )

            st.download_button(
                label="⬇️ Download Insight Report",
                data=report_download,
                file_name="insightforge_report.md",
                mime="text/markdown"
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