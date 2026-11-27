# FinWise AI – AI-powered Personal Finance Manager

A Streamlit app that tracks income and expenses, auto-categorises transactions with machine learning,
flags unusual spending, forecasts next month's spending and gives personalised savings tips.
Sample data (6 months of an Indian user's transactions, in ₹) loads automatically, so the app works
straight away. It runs fully offline and needs no API keys.

## Requirements

- Python 3.10 or newer
- The five packages in `requirements.txt`: streamlit, pandas, numpy, scikit-learn, plotly

## Setup

Open a terminal in the `finwise-ai` folder.

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (cmd): `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

Install the packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python data/generate_data.py   # (optional) rebuild the sample CSVs - same files every time
python smoke_test.py           # trains every model and checks the outputs (prints the accuracy)
streamlit run app.py           # opens the app at http://localhost:8501
```

On Windows without activating the environment you can call it directly, e.g.
`.venv\Scripts\python.exe smoke_test.py` and `.venv\Scripts\streamlit.exe run app.py`.

## Pages

| Page | What it shows |
|---|---|
| Dashboard | Total income, expenses, savings and savings rate; monthly income vs expenses; category breakdown |
| Add Transaction | Type a description and the AI predicts its category live, with confidence; you can override it |
| Upload CSV | Upload your own CSV; rows without a category are categorised by the model |
| Spending Patterns | Month-over-month change per category, top merchants, weekday vs weekend spending |
| Anomalies | Unusual expenses, each with a plain-English reason ("12.8× what you usually pay at Uber") |
| Forecast | Next month's predicted spend per category next to the actuals, plus an accuracy test |
| Recommendations | 5–8 tips with rupee amounts, the 50/30/20 check and detected subscriptions |
| AI Model | Test accuracy, confusion matrix and mistakes of the categoriser |

Use **Reset to sample data** in the sidebar to undo added or uploaded rows.

## CSV format for uploads

```
date,description,amount,type,category
2026-10-26,Chai Point - Masala chai and samosa,140,expense,
31/10/2026,Fiverr - Logo design payout,4200,income,Other Income
```

- `date`: `2026-10-31` or `31/10/2026`
- `amount`: a number (negative values are turned positive)
- `type`: `income` or `expense` (`credit` / `debit` also work)
- `category`: optional - leave it empty and the AI fills it in

The Upload page has a button that downloads an example file in this format.

## Project structure

```
finwise-ai/
├── app.py                  Streamlit app (all pages)
├── ai/
│   ├── categorizer.py      TF-IDF + Logistic Regression on descriptions
│   ├── anomaly.py          Isolation Forest for unusual expenses
│   ├── forecast.py         Linear Regression next-month forecast per category
│   └── recommender.py      spending analysis + personalised savings tips
├── data/
│   ├── generate_data.py    deterministic sample-data generator (seed 42)
│   ├── transactions.csv    435 demo transactions, May–Oct 2026
│   └── training_data.csv   720 labelled descriptions for the categoriser
├── smoke_test.py           end-to-end checks of every AI module
├── requirements.txt
├── README.md
└── REPORT.md               project report
```

All randomness uses fixed seeds, so the data, the models and every number are the same on every run.
