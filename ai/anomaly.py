"""
Anomaly detection: finds expenses that are unusual *for this user*.

Algorithm: Isolation Forest (scikit-learn). It builds many random trees that
keep cutting the data at random points. Normal expenses sit in crowded areas
and need many cuts before they are alone; an unusual expense is isolated after
only a few cuts. A short average path length means "anomaly".

Features - both measure how big a payment is compared with what is normal for you:
  1. category ratio - amount / your median amount in that category
  2. merchant ratio - amount / your median amount at that merchant
     (falls back to the category median if you paid that merchant fewer than 3 times)

Contamination = 0.02 tells the forest to flag the most unusual 2% of expenses.

We take the log of each ratio and clip negative values to 0, because paying
*less* than usual is never a problem. Using ratios instead of raw rupees means
rent (big, but normal for rent) is not flagged, while a Rs 2,350 Uber ride
(about 10x your usual ride) is.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ai.categorizer import extract_merchant
from ai.recommender import format_inr

SEED = 42
CONTAMINATION = 0.02       # we expect roughly 2 in every 100 expenses to be unusual
MIN_MERCHANT_COUNT = 3     # need this many payments before a merchant has its own "normal"
MIN_EXPENSES = 20          # too little data to say what is unusual below this


def add_features(expenses):
    """Add the merchant name, the 'typical' amounts and the two ratio features."""
    exp = expenses.copy()
    exp["merchant"] = exp["description"].map(extract_merchant)
    merchant_key = exp["merchant"].str.lower()
    exp["category_median"] = exp.groupby("category")["amount"].transform("median")
    exp["merchant_count"] = exp.groupby(merchant_key)["amount"].transform("size")
    exp["merchant_median"] = exp.groupby(merchant_key)["amount"].transform("median")
    few = exp["merchant_count"] < MIN_MERCHANT_COUNT
    exp.loc[few, "merchant_median"] = exp.loc[few, "category_median"]
    exp["category_ratio"] = exp["amount"] / exp["category_median"]
    exp["merchant_ratio"] = exp["amount"] / exp["merchant_median"]
    return exp


def feature_matrix(exp):
    """The numbers the Isolation Forest sees: log of both ratios, never below 0."""
    return np.clip(np.log(exp[["category_ratio", "merchant_ratio"]].to_numpy()), 0, None)


def explain(row):
    """Plain-English reason why one flagged expense looks unusual."""
    category_reason = (f"{row['category_ratio']:.1f}× your typical {row['category']} spend "
                       f"(usually {format_inr(row['category_median'])})")
    reasons = [category_reason]
    if row["merchant_count"] >= MIN_MERCHANT_COUNT and row["merchant_ratio"] >= 1.5:
        merchant_reason = (f"{row['merchant_ratio']:.1f}× what you usually pay at {row['merchant']} "
                           f"({format_inr(row['merchant_median'])})")
        if row["merchant_ratio"] > row["category_ratio"]:
            reasons.insert(0, merchant_reason)
        else:
            reasons.append(merchant_reason)
    elif row["merchant_count"] == 1:
        reasons.append(f"first payment to {row['merchant']}")
    return "; ".join(reasons)


def detect_anomalies(df, contamination=CONTAMINATION):
    """Score every expense and flag the unusual ones.

    Returns all expenses (same index as `df`) with extra columns:
      is_anomaly    - True for flagged expenses
      anomaly_score - higher = more unusual
      reason        - plain-English explanation (only for flagged rows)
    """
    expenses = df[(df["type"] == "expense") & (df["amount"] > 0)]
    if len(expenses) < MIN_EXPENSES:
        return expenses.assign(is_anomaly=False, anomaly_score=0.0, reason="")

    exp = add_features(expenses)
    X = feature_matrix(exp)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED)
    exp["is_anomaly"] = forest.fit_predict(X) == -1
    exp["anomaly_score"] = -forest.score_samples(X)
    exp["reason"] = ""
    flagged = exp["is_anomaly"]
    if flagged.any():
        exp.loc[flagged, "reason"] = exp[flagged].apply(explain, axis=1)
    return exp
