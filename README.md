# 🔎 InsightForge

### AI Data Detective — Discover what your data is trying to tell you.

InsightForge is an AI-powered data investigation tool that automatically analyzes CSV and Excel datasets to discover trends, anomalies, data-quality issues, and interesting patterns.

Instead of simply asking an AI questions about your data, InsightForge first **detects something interesting** and then lets you investigate it.

---

## 🚀 What InsightForge Does

Upload a CSV or Excel file and InsightForge automatically:

- 📊 Profiles the dataset
- 🔍 Analyzes columns and data types
- ⚠️ Detects missing values
- 🔁 Detects duplicate rows
- 📈 Generates statistical summaries
- 📉 Detects basic trends
- 🚨 Detects statistical anomalies
- 📊 Generates interactive visualizations
- 🕵️ Investigates individual anomalies
- 🤖 Uses Gemma to explain findings
- 💡 Generates an AI-powered data investigation report
- ⬇️ Allows report and dataset export

---

## 🕵️ The Key Idea

Most AI data-analysis tools follow this pattern:

> **Ask a question → AI answers**

InsightForge follows a different approach:

> **Detect something interesting → Investigate it → Understand why it matters**

This makes the system useful even when the user does not know what questions to ask.

---

## 🧠 How It Works

```text
             CSV / Excel
                  │
                  ▼
          ┌─────────────────┐
          │  Data Profiling │
          └────────┬────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Detection Engine    │
        │                     │
        │ • Data Quality      │
        │ • Trends            │
        │ • Anomalies         │
        │ • Statistics        │
        └──────────┬──────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Investigate This│
          └────────┬────────┘
                   │
                   ▼
             ┌───────────┐
             │  Gemma 4  │
             └─────┬─────┘
                   │
                   ▼
        Evidence + Explanation
                   │
                   ▼
             AI Report
