# 🔍 InsightForge

### AI Data Detective — Discover what your data is trying to tell you.

InsightForge is an AI-powered data analysis tool that automatically explores datasets, detects trends and anomalies, checks data quality, and uses **Gemma 4** to help users investigate interesting findings.

Instead of requiring users to know exactly what question to ask, InsightForge first looks at the data and identifies potentially meaningful patterns.

---

## 🚀 Why InsightForge?

Most data-analysis tools follow this workflow:

> Ask a question → Analyze the data → Get an answer

InsightForge takes a different approach:

> Upload data → Discover something interesting → Investigate it with AI

The goal is to help users discover insights they may not have thought to look for.

---

## ✨ Key Features

### 📊 Dataset Overview
- Upload CSV or Excel files
- View dataset dimensions
- Preview uploaded data
- Automatically identify numeric and categorical columns

### 🧹 Data Quality Analysis
- Detect missing values
- Detect duplicate rows
- Show column data types
- Display basic statistical information

### 📈 Automatic Trend Detection
InsightForge automatically searches for:
- Date/time columns
- Numeric columns
- Changes between the beginning and end of a dataset

It then generates an interactive Plotly chart showing the detected trend.

### 🚨 Smart Anomaly Detection

InsightForge combines statistical techniques to identify potentially unusual values:

- IQR-based anomaly detection
- Z-score analysis
- Numeric-column analysis
- Highlighting potentially unusual observations

This helps users find values that deserve further investigation.

### 🕵️ Investigate This

This is the core feature of InsightForge.

When an interesting anomaly is detected, users can investigate it using Gemma.

Python first calculates and verifies the relevant evidence.

Gemma then interprets that evidence and explains:

- What happened
- Why it may be interesting
- What patterns are visible
- What users should investigate next

This separation helps reduce unsupported AI-generated conclusions.

### 💡 Highest Value Investigation

Users can select a numeric column and investigate the row containing its highest value.

InsightForge provides the relevant dataset evidence to Gemma for interpretation.

### 🤖 AI Data Detective

Gemma is used as an analytical assistant rather than as the primary calculator.

The application follows this principle:

> **Python calculates the facts. Gemma explains the facts.**

### 📄 AI Insight Report

InsightForge can generate a structured AI report containing:

- Dataset overview
- Data quality observations
- Trends
- Potential anomalies
- Important findings
- Suggested areas for investigation

The report can be downloaded as a Markdown file.

### 💾 Data Export

Users can download:

- The generated AI report
- The uploaded dataset
LINK: https://insightforge-2grbxdarbfpbe6kdp6skdc.streamlit.app/
---

## 🧠 How It Works

```text
                 ┌──────────────────────┐
                 │     Upload Dataset   │
                 │      CSV / Excel     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Data Profiler     │
                 │                      │
                 │ • Shape              │
                 │ • Data Types         │
                 │ • Missing Values     │
                 │ • Duplicates         │
                 │ • Statistics         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Discovery Engine    │
                 │                      │
                 │ • Trends             │
                 │ • Anomalies          │
                 │ • Data Quality       │
                 │ • Distributions      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Investigate This  │
                 │                      │
                 │  Verified evidence   │
                 │          +           │
                 │       Gemma 4        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      AI Insights     │
                 │                      │
                 │ • Explanation       │
                 │ • Findings          │
                 │ • Investigation     │
                 │   suggestions       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       AI Report      │
                 └──────────────────────┘
