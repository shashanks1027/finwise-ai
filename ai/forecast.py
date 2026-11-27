"""
Forecast: predicts next month's spending in every category.

Algorithm: Linear Regression (scikit-learn), one small model per category.
  Input  X = month number (0 for the first month, 1 for the next, ...)
  Output y = total spent in that category in that month
The model finds the straight line that best follows the monthly totals and
extends it one month ahead. The slope of the line is the trend in rupees per month.

One-off spikes found by the anomaly detector can be left out first
(`exclude_index`), so a single big purchase does not bend the line.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def monthly_totals(df, exclude_index=None):
    """Table of total expense per month (rows) and category (columns). Missing months count as 0."""
    expenses = df[df["type"] == "expense"]
    if exclude_index is not None:
        expenses = expenses.drop(index=exclude_index, errors="ignore")
    months = expenses["date"].dt.to_period("M").rename("month")
    table = expenses.pivot_table(index=months, columns="category", values="amount",
                                 aggfunc="sum", fill_value=0)
    all_months = pd.period_range(table.index.min(), table.index.max(), freq="M")
    return table.reindex(all_months, fill_value=0).astype(float)


def fit_line(values):
    """Fit y = a + b*x to the monthly values. Returns (next month's prediction, slope b)."""
    if len(values) < 2:
        return float(values[-1]), 0.0
    X = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression().fit(X, values)
    prediction = model.predict(np.array([[len(values)]]))[0]
    return max(0.0, float(prediction)), float(model.coef_[0])


def forecast_next_month(df, exclude_index=None):
    """Forecast every category for the month after the data ends.

    Returns (forecast table, monthly totals table, name of the forecast month).
    """
    table = monthly_totals(df, exclude_index)
    rows = []
    for category in table.columns:
        values = table[category].to_numpy()
        prediction, slope = fit_line(values)
        rows.append({"category": category, "last_month": values[-1], "average": values.mean(),
                     "forecast": prediction, "trend_per_month": slope})
    forecast = pd.DataFrame(rows).sort_values("forecast", ascending=False).reset_index(drop=True)
    next_month = (table.index[-1] + 1).strftime("%b %Y")
    return forecast, table, next_month


def backtest(df, exclude_index=None, test_months=3):
    """Check the forecaster honestly on months it has not seen.

    For each of the last `test_months` months we train only on the months
    before it, predict that month's total spending, and compare with what
    really happened (a "rolling-origin" test). A simple baseline - the average
    of the earlier months - is tested the same way for comparison.

    Returns one row per test month, or None if there is too little data
    (each test needs at least 2 earlier months).
    """
    table = monthly_totals(df, exclude_index)
    first_test = max(2, len(table) - test_months)
    if first_test >= len(table):
        return None
    rows = []
    for k in range(first_test, len(table)):
        history, actual = table.iloc[:k], table.iloc[k].sum()
        predicted = sum(fit_line(history[c].to_numpy())[0] for c in table.columns)
        baseline = history.mean().sum()
        rows.append({"month": table.index[k].strftime("%b %Y"), "actual": actual,
                     "linear_regression": predicted, "average_baseline": baseline,
                     "lr_error_pct": abs(predicted - actual) / actual * 100,
                     "baseline_error_pct": abs(baseline - actual) / actual * 100})
    return pd.DataFrame(rows)
