# InsightForge

**AI Data Detective --- discover what your data is trying to tell you.**

InsightForge is an AI-assisted data exploration application that helps
users inspect CSV and Excel datasets, identify potentially meaningful
patterns, investigate anomalies, and communicate findings through
natural-language explanations and downloadable reports.

**Live application:**
https://insightforge-n9kaj2fkhfqzb78xeptfdz.streamlit.app\
**Track:** Track 1 → G5: Open Innovation with Gemma 4
And find a sample dataset above👆
## Overview

Many data workflows require users to know what to look for before they
begin. InsightForge takes a discovery-first approach: it profiles a
dataset, surfaces patterns worth investigating, and gives users tools to
explore those findings.

The application combines deterministic Python-based data analysis with a
generative model for explanations. Python computes the underlying
statistics; the model helps translate selected findings into accessible
language. AI-generated explanations should be treated as assistance,
while numeric results should be checked against the source data.

## Core capabilities

-   **Dataset ingestion:** Upload CSV and Excel files for analysis.
-   **Data profiling:** Review dataset structure, columns, data types,
    and quality indicators.
-   **Trend and pattern exploration:** Examine distributions,
    relationships, correlations, and trends in the available data.
-   **Anomaly detection:** Surface unusual observations using
    statistical techniques such as IQR and Z-score where applicable.
-   **Investigate findings:** Explore selected observations and request
    an AI-generated explanation.
-   **Ask Your Data:** Ask natural-language questions about a dataset
    and receive AI-assisted responses.
-   **Visual analytics:** Explore findings through interactive charts
    and dashboard views.
-   **Reporting:** Generate downloadable PDF reports to communicate
    results.

Feature availability and output quality depend on the dataset,
configured model access, and the current application build.

## Innovation

InsightForge focuses on integrating several steps into one workflow:

1.  Inspect and profile a dataset.
2.  Surface potential data-quality issues, trends, and anomalies.
3.  Let the user investigate findings rather than requiring a fully
    specified question at the start.
4.  Use a generative model to explain selected results in natural
    language.
5.  Consolidate findings into a report.

The contribution is the integrated, discovery-first workflow---not a
claim that each individual capability is new. Unlike a
visualization-only workflow, InsightForge is designed to connect
statistical exploration with conversational interpretation and
reporting.

## Track alignment

**Track 1 → G5: Open Innovation with Gemma 4**

InsightForge explores how Gemma 4 can make data analysis more accessible
by helping users interpret statistical findings through natural-language
explanations. The application pairs conventional Python data processing
with model-assisted interpretation, keeping numeric analysis separate
from generated explanations.

## Technical architecture

``` text
CSV / Excel dataset
        |
        v
Streamlit application interface
        |
        v
Python data-processing and analysis layer
(Pandas, NumPy, statistical checks)
        |
        +------------------------------+
        |                              |
        v                              v
Charts and structured results     Selected findings
(Plotly)                               |
                                       v
                              Google GenAI SDK
                              Gemma 4 / configured model
                                       |
                                       v
                              Natural-language explanation
        |
        v
Dashboard views and PDF reporting
```

The application uses a Python-driven analysis workflow. It does not
require LangGraph for the architecture described here; do not describe
it as a LangGraph agent unless that framework is added to the
implementation.

## Technology stack

  -----------------------------------------------------------------------
  Component               Technology              Responsibility
  ----------------------- ----------------------- -----------------------
  Application interface   Streamlit               User interface and
                                                  interactive workflow

  Data processing         Python, Pandas, NumPy   Dataset inspection and
                                                  statistical analysis

  Visualization           Plotly                  Interactive charts

  Generative AI           Google GenAI SDK        Model requests for
  integration                                     AI-assisted
                                                  explanations

  AI model                Gemma 4 / configured    Natural-language
                          Google model            interpretation

  Excel support           OpenPyXL                Excel file handling
                                                  where required

  PDF reporting           ReportLab               PDF report generation

  Deployment              Streamlit Community     Hosted application
                          Cloud                   
  -----------------------------------------------------------------------

## Getting started

### Requirements

-   Python 3.10 or a version supported by the dependencies in
    `requirements.txt`
-   A Google GenAI API key for features that call the configured model
-   Git

### Install

``` bash
git clone https://github.com/AkifAhmedShariff/insightforge.git
cd insightforge

python -m venv .venv
```

Activate the environment:

**macOS / Linux**

``` bash
source .venv/bin/activate
```

**Windows**

``` powershell
.venv\Scripts\activate
```

Install dependencies and launch the application:

``` bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Configuration

For local development, place secrets in a `.env` file only if the
application code is configured to load it. Do not commit API keys.

For Streamlit Community Cloud, add the required secret in the app's
**Settings → Secrets** section:

``` toml
GEMINI_API_KEY = "your_api_key_here"
```

Use the key name expected by the current application code. Never publish
real credentials in source control, README examples, screenshots, or
issue reports.

## Repository structure

The repository includes the following core files and directories:

``` text
insightforge/
├── app.py
├── requirements.txt
├── sample_sales.csv
├── LICENSE
├── screenshots/
└── README.md
```

The exact file list may evolve as the project develops.

## Responsible use and limitations

-   An anomaly is a signal for investigation, not proof of an error or
    fraud.
-   Statistical methods can miss relevant patterns or flag valid
    observations.
-   Model-generated explanations can be incomplete or incorrect; verify
    important claims against the underlying data.
-   Results depend on input quality, dataset size, available columns,
    and model configuration.
-   Do not upload confidential or personally identifiable data unless
    you are authorized to do so and the deployment is appropriate for
    that data.

## Contributing

Contributions, issue reports, and practical suggestions are welcome.
When reporting a bug, include the steps to reproduce it and a sanitized
error message. Never include API keys, passwords, or private datasets.

## License

See [`LICENSE`](LICENSE) for the terms governing use and distribution.
