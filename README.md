# Procurement Fraud & Cost Anomaly Detection Engine

> **Identifying high-risk transactions in procurement datasets using unsupervised machine learning, statistical scoring, and rule-based pattern detection.**

---

## Overview

Organisations processing thousands of procurement invoices annually face material exposure to duplicate billing, spend spikes, and vendor concentration risk — risks that manual audit processes routinely miss. This project provides a reproducible, configurable anomaly detection pipeline designed for financial audit teams and data analytics practitioners operating in regulated environments.

The engine combines three complementary detection methods and surfaces results through a production-ready analytical dashboard.

---

## Detection Methodology

| Layer | Method | Signal |
|---|---|---|
| Multivariate | Isolation Forest (scikit-learn) | Unusual combinations of amount, frequency, vendor behaviour |
| Univariate | Z-score threshold (σ > 2.5) | Spend spikes relative to dataset distribution |
| Rule-based | Deterministic flags | Duplicate invoices · Round-sum bias · New vendor large payments |

Each flagged transaction receives a continuous **anomaly score** and is assigned a **risk tier** (High / Medium / Low) to support triage and prioritisation by audit teams.

---

## Features

- **Interactive dashboard** — timeline scatter, vendor concentration chart, category heatmap, detection signal breakdown
- **Configurable sensitivity** — adjust the model contamination rate via the sidebar
- **CSV upload** — plug in your own transaction data with minimal reformatting
- **Export** — download full scored results as CSV for further analysis
- **Synthetic data generator** — generates 500+ realistic procurement transactions with seeded anomaly types for reproducible testing

---

## Anomaly Types Detected

- `duplicate_invoice` — same vendor, same amount billed more than once
- `amount_spike` — transaction value significantly above vendor/category baseline
- `round_sum_bias` — suspiciously round invoice amounts (e.g. exactly £25,000)
- `high_frequency_billing` — unusually short interval between invoices from same vendor
- `new_vendor_large_payment` — large payment to a vendor with no prior transaction history

---

## Project Structure

```
procurement-anomaly-engine/
│
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── src/
│   ├── detector.py           # Core ML pipeline (IsolationForest + feature engineering)
│   └── data_generator.py     # Synthetic procurement data generator
│
├── tests/
│   └── test_detector.py      # Pytest unit tests (10 test cases)
│
└── data/
    └── sample_transactions.csv   # Pre-generated sample data (optional)
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/procurement-anomaly-engine.git
cd procurement-anomaly-engine
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Run tests
pytest tests/ -v
```

---

## Input Data Schema

If uploading your own CSV, the following columns are required:

| Column | Type | Example |
|---|---|---|
| `invoice_id` | string | `INV-10042` |
| `invoice_date` | date (ISO 8601) | `2024-03-15` |
| `vendor_id` | string | `V001` |
| `vendor_name` | string | `Apex Construction Ltd` |
| `category` | string | `Construction` |
| `department` | string | `Finance` |
| `amount` | float | `12500.00` |

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `pandas` · `NumPy` · `Streamlit` · `Plotly` · `pytest`

---

## Relevance to Financial Analytics

This project reflects the analytical challenges encountered in financial audit roles — specifically:

- Designing detection logic for datasets exceeding £10M in transactional value
- Applying unsupervised ML where labelled fraud data is unavailable
- Surfacing actionable insights to non-technical stakeholders via dashboard tooling
- Implementing explainability (feature contribution analysis) to support audit defence

---

## Author

**Malavika Sudheer** · Data Analyst · MSc Data Science, University of Essex  
[linkedin.com/in/malavika-sudheer24](https://linkedin.com/in/malavika-sudheer24)
