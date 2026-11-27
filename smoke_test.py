"""
FinWise AI - smoke test
=======================
Trains every model, runs every AI module on the sample data and checks that
the outputs make sense. The numbers it prints are the ones quoted in REPORT.md.

Run from the finwise-ai folder:
    python smoke_test.py
"""

import io
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ai.anomaly import CONTAMINATION, detect_anomalies
from ai.categorizer import SEED, accuracy_on, categorize_missing, clean_text, predict_categories, train_categorizer
from ai.forecast import backtest, forecast_next_month
from ai.recommender import (detect_subscriptions, generate_recommendations, month_over_month,
                            needs_wants_savings, overall_totals, prepare_transactions, top_merchants,
                            weekday_weekend_split)
from data.generate_data import (PLANTED_ANOMALIES, TRAINING_CSV, TRANSACTIONS_CSV, build_training_data,
                                build_transactions, main as write_sample_data)

sys.stdout.reconfigure(encoding="utf-8")  # so ₹ and × print on any Windows terminal
checks_passed = 0


def check(condition, message):
    """Stop with an error if `condition` is false; otherwise print a tick."""
    global checks_passed
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    checks_passed += 1
    print(f"  [ok] {message}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
section("1. Sample data")
if not TRANSACTIONS_CSV.exists() or not TRAINING_CSV.exists():
    write_sample_data()
check(build_transactions().equals(build_transactions()), "transaction generator is deterministic")
check(build_training_data().equals(build_training_data()), "training-data generator is deterministic")
on_disk = TRANSACTIONS_CSV.read_text(encoding="utf-8")
check(on_disk == build_transactions().to_csv(index=False, lineterminator="\n"),
      "transactions.csv matches a fresh run of generate_data.py")
df = prepare_transactions(pd.read_csv(TRANSACTIONS_CSV))
training = pd.read_csv(TRAINING_CSV)
check(400 <= len(df) <= 600, f"transactions.csv has {len(df)} rows (400-600)")
check(len(training) >= 600, f"training_data.csv has {len(training)} labelled descriptions (600+)")
print(f"  period: {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}, "
      f"{df['category'].nunique()} categories")

# ---------------------------------------------------------------------------
section("2. AI categorizer (TF-IDF + Logistic Regression)")
result = train_categorizer()
model = result["model"]
print(f"  train / test split: {result['n_train']} / {result['n_test']} descriptions")
print(f"  TEST ACCURACY (held-out 20%): {result['accuracy']:.1%}")
check(result["accuracy"] >= 0.85, "test accuracy is at least 85%")
demo_accuracy = accuracy_on(model, df)
print(f"  accuracy on the {len(df)} demo transactions (never used for training): {demo_accuracy:.1%}")
print(f"  macro-average F1 on the test set: {result['report']['f1-score'].mean():.3f}")
print(f"  test mistakes: {len(result['mistakes'])}")
for row in result["mistakes"].itertuples():
    print(f"     '{row.description}'  actual={row.actual}  predicted={row.predicted}")

new_descriptions = {
    "Swiggy order 5521": "Food & Dining",
    "Ola cab to office": "Transport",
    "Netflix renewal": "Entertainment",
    "Salary for November - Acme": "Salary",
    "BigBasket weekly groceries": "Groceries",
    "Jio recharge 666": "Bills & Utilities",
    "Paid to landlord - rent Nov": "Rent",
    "Apollo Pharmacy medicines": "Health",
}
predictions = predict_categories(model, new_descriptions.keys())
correct = 0
for (text, expected), row in zip(new_descriptions.items(), predictions.itertuples()):
    correct += row.category == expected
    print(f"     '{text}' -> {row.category} ({row.confidence:.0%})")
check(correct >= 7, f"{correct}/8 brand-new descriptions categorised correctly")

blank = df.assign(category=pd.NA)
filled = categorize_missing(model, blank)
check(filled["category"].notna().all() and (filled["categorized_by"] == "AI").all(),
      "categorize_missing fills every missing category")
check(filled["confidence"].between(0, 1).all(), "every AI confidence is between 0 and 1")

# Why character n-grams? Compare them with word n-grams using 5-fold cross-validation.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, vectorizer in [
    ("character 2-4-grams (used)", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                                   ngram_range=(2, 4), sublinear_tf=True)),
    ("word 1-2-grams", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
]:
    pipe = Pipeline([("tfidf", vectorizer), ("model", LogisticRegression(C=10, max_iter=2000, random_state=SEED))])
    scores = cross_val_score(pipe, training["description"], training["category"], cv=folds)
    print(f"  5-fold CV accuracy, {name}: {scores.mean():.1%}")

# ---------------------------------------------------------------------------
section("3. Anomaly detection (Isolation Forest)")
scored = detect_anomalies(df)
flagged = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)
print(f"  expenses checked: {len(scored)}, contamination: {CONTAMINATION:.0%}, flagged: {len(flagged)}")
for row in flagged.itertuples():
    print(f"     {row.date:%d %b}  {format(row.amount, ',.0f'):>7}  {row.description}  -> {row.reason}")
check(1 <= len(flagged) <= 0.10 * len(scored), "between 1 and 10% of expenses are flagged")
check((flagged["reason"].str.contains("×")).all(), "every flagged expense has a plain-English reason")
check(not (flagged["category"] == "Rent").any(), "rent (large but normal) is not flagged")
planted = {text for _, text, _, _ in PLANTED_ANOMALIES}
caught = planted & set(flagged["description"])
print(f"  PLANTED ANOMALIES CAUGHT: {len(caught)} of {len(planted)}")
check(len(caught) >= 5, "at least 5 of the 7 planted anomalies are caught")

# ---------------------------------------------------------------------------
section("4. Forecast (Linear Regression per category)")
exclude = scored.index[scored["is_anomaly"]]
forecast, table, next_month = forecast_next_month(df, exclude_index=exclude)
print(f"  forecast month: {next_month} (trained on {len(table)} months, anomalies left out)")
for row in forecast.itertuples():
    print(f"     {row.category:<18} last {row.last_month:>8,.0f}  avg {row.average:>8,.0f}  "
          f"forecast {row.forecast:>8,.0f}  trend {row.trend_per_month:+,.0f}/month")
check((forecast["forecast"] >= 0).all(), "no negative forecasts")
check(next_month == "Nov 2026", "forecast is for the month after the data ends")
print(f"  total forecast for {next_month}: {forecast['forecast'].sum():,.0f}")
bt = backtest(df, exclude_index=exclude)
for row in bt.itertuples():
    print(f"     backtest {row.month}: actual {row.actual:,.0f} | LR {row.linear_regression:,.0f} "
          f"({row.lr_error_pct:.1f}% off) | average {row.average_baseline:,.0f} ({row.baseline_error_pct:.1f}% off)")
print(f"  BACKTEST MEAN ERROR: Linear Regression {bt['lr_error_pct'].mean():.1f}% | "
      f"6-month-average baseline {bt['baseline_error_pct'].mean():.1f}%")
check(len(bt) == 3, "backtest ran on 3 held-out months")

# ---------------------------------------------------------------------------
section("5. Spending analysis")
totals = overall_totals(df)
print(f"  income {totals['income']:,.0f} | expenses {totals['expense']:,.0f} | "
      f"savings {totals['savings']:,.0f} | savings rate {totals['savings_rate']:.1f}%")
nws = needs_wants_savings(df)
print(f"  50/30/20: needs {nws['needs_pct']:.1f}% | wants {nws['wants_pct']:.1f}% | savings {nws['savings_pct']:.1f}%")
mom = month_over_month(df)
check(not mom.empty, f"month-over-month table built ({mom.attrs['months'][0]} -> {mom.attrs['months'][1]})")
merchants = top_merchants(df, n=5)
print("  top merchants: " + ", ".join(f"{r.merchant} ({r.total:,.0f})" for r in merchants.itertuples()))
check(len(merchants) == 5, "top merchants found")
split = weekday_weekend_split(df)
print(f"  spend per weekday {split['per_weekday'].sum():,.0f} vs per weekend day {split['per_weekend_day'].sum():,.0f}")
subs = detect_subscriptions(df)
print("  recurring payments: " + ", ".join(f"{r.name} ({r.amount:,.0f})" for r in subs.itertuples()))
check("Netflix" in set(subs["name"]), "Netflix detected as a recurring subscription")

# ---------------------------------------------------------------------------
section("6. Recommendations")
tips = generate_recommendations(df, scored)
for i, t in enumerate(tips, 1):
    print(f"  {i}. {t['title']}  [saves {t['monthly_saving']:,.0f}/month]\n     {t['detail']}")
check(5 <= len(tips) <= 8, f"{len(tips)} recommendations (5-8)")
check(all("₹" in t["title"] + t["detail"] for t in tips), "every tip quotes a rupee amount")

# ---------------------------------------------------------------------------
section("7. CSV upload path")
upload = io.StringIO(
    "date,description,amount,type,category\n"
    "03/11/2026,Zomato - Order #5521,-420,expense,\n"
    "2026-11-04,UPI/Rapido/883920112,95,debit,\n"
    "05/11/2026,Netflix - Monthly subscription,649,expense,Entertainment\n"
    "2026-11-01,Salary Credit - Acme Technologies Pvt Ltd,78000,income,\n")
uploaded = categorize_missing(model, prepare_transactions(pd.read_csv(upload)))
check(len(uploaded) == 4 and uploaded["category"].notna().all(), "uploaded rows are cleaned and categorised")
check(uploaded["date"].min() == pd.Timestamp("2026-11-01")
      and uploaded["date"].max() == pd.Timestamp("2026-11-05"), "Indian dd/mm/yyyy dates are read correctly")
check((uploaded["amount"] > 0).all() and set(uploaded["type"]) == {"income", "expense"},
      "negative amounts and 'debit' are normalised")
print(uploaded[["date", "description", "category", "confidence", "categorized_by"]].to_string(index=False))
all_blank = io.StringIO("date,description,amount,type,category\n"
                        "2026-11-02,Zepto - Daily essentials,386,expense,\n"
                        "2026-11-03,PVR Cinemas - 2 tickets,720,expense,\n")
blank_upload = categorize_missing(model, prepare_transactions(pd.read_csv(all_blank)))
check(blank_upload["category"].notna().all(), "a CSV whose category column is completely empty is categorised")
typed = categorize_missing(model, df.assign(category=pd.NA))
income_cats = typed["category"].isin(["Salary", "Other Income"])
check((income_cats == (typed["type"] == "income")).all(),
      "AI never gives an expense an income category (or the other way round)")
print(f"  accuracy on demo transactions when the type is used: {(typed['category'] == df['category']).mean():.1%}")

print(f"\nALL {checks_passed} CHECKS PASSED")
