# 🔍 InsightForge

### AI Data Detective — Discover what your data is trying to tell you.

InsightForge is an AI-powered data analysis application built with Python and Streamlit. It helps users explore datasets, identify trends and anomalies, investigate findings using AI, ask questions in natural language, and generate downloadable reports.

Instead of requiring users to perform every analysis manually, InsightForge combines Python-based calculations with AI-assisted interpretation to make data exploration more accessible.

**Live Demo:** [Try InsightForge](NA LINK)

---

## ✨ Features

### 📊 Dataset Exploration
- Upload CSV and Excel datasets.
- Preview data and inspect dataset dimensions.
- Identify numeric and categorical columns.
- Review descriptive statistics and column data types.

### 🧹 Data Quality Analysis
- Identify missing values.
- Detect duplicate records.
- Inspect column types and summary statistics.
- Understand potential data-quality issues before analysis.

### 📈 Automatic Trend Discovery
- Identify potential date and numeric columns.
- Explore changes and patterns in the data.
- Visualize findings with interactive charts.

### 🚨 Anomaly Detection
- Use statistical techniques such as the Interquartile Range (IQR) and Z-scores.
- Highlight potentially unusual observations.
- Investigate interesting values using relevant dataset evidence.

### 💬 Ask Your Data
Ask questions about an uploaded dataset using natural language rather than writing queries manually.

The analysis workflow combines structured analysis planning, Python-based calculations, and AI-assisted explanations to help users explore data and interpret results.

### 🤖 AI Data Detective
InsightForge uses AI to help explain patterns and suggest further investigations.

> **Core principle:** Python calculates the evidence; AI helps explain what it may mean.

AI-generated explanations should be interpreted in the context of the underlying data and statistical results.

### 📄 PDF Report Generation
Generate downloadable PDF reports containing supported report sections, charts, and narrative content.

The report-generation workflow is designed to make analytical findings easier to review and share.

### 💾 Data Export
Export available analysis outputs and datasets for further review or reporting.

---

## 🧠 How It Works

1. **Upload** — Load a CSV or Excel dataset.
2. **Profile** — Inspect dimensions, column types, statistics, and data quality.
3. **Discover** — Explore trends, distributions, and potential anomalies.
4. **Investigate** — Ask questions or select interesting findings for deeper analysis.
5. **Interpret** — Use AI-assisted explanations grounded in computed evidence.
6. **Report** — Generate and download available reports.

---

## 🛠️ Technology Stack

- **Python** — Data processing and analytical logic
- **Streamlit** — Interactive web application
- **Pandas** — Data manipulation and analysis
- **Plotly** — Interactive visualizations
- **Gemma / Gemini API integration** — AI-assisted interpretation
- **ReportLab** — PDF report generation

---

## 🚀 Getting Started

### Prerequisites

- Python 3
- Git
- An API key configured for the AI integration, if using AI-powered features

### 1. Clone the repository

```bash
git clone https://github.com/AkifAhmedShariff/insightforge.git
cd insightforge
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file and configure the API key using the exact environment variable expected by the application.

Never commit `.env` or expose API keys in source code, screenshots, logs, or public commits.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in your terminal.

---

## 🧪 Validation

Check Python syntax with:

```bash
python3 -m py_compile app.py
```

Check Git whitespace errors with:

```bash
git diff --check
```

These checks are useful initial validations; they do not replace application runtime tests or end-to-end testing.

---

## 📁 Project Structure

```text
insightforge/
├── app.py                  # Streamlit application
├── data_analysis.py        # Data-analysis helpers
├── tests/
│   └── test_data_analysis.py
├── requirements.txt        # Python dependencies
├── sample_sales.csv        # Sample dataset
├── screenshots/            # Project screenshots
└── README.md               # Project documentation
```

The exact files available may vary by branch.

---

## 🔐 Security Notes

- Keep API keys in local environment configuration.
- Never commit `.env` files or credentials.
- Review uploaded data before sharing reports publicly.
- Verify important AI-generated interpretations against the underlying data.

---

## 🗺️ Roadmap

- Expand automated data-analysis workflows.
- Improve natural-language question handling.
- Enhance report formatting and export options.
- Add more comprehensive automated and end-to-end tests.
- Improve usability and documentation based on feedback.

---

## 🤝 Contributing

Contributions, suggestions, bug reports, and feature ideas are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes and push your branch.
5. Open a pull request describing the changes.

---

## 📄 License

See the [`LICENSE`](LICENSE) file for licensing information.

---

**InsightForge — Turn raw data into meaningful discoveries.**
