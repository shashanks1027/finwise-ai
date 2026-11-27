"""
Spending analysis and personalised savings recommendations.

Part 1 - analysis helpers (plain pandas): cleaning the data, monthly summary,
month-over-month change, top merchants, weekday vs weekend split, the
50/30/20 split and recurring-subscription detection.

Part 2 - generate_recommendations(): a rule-based engine that turns those
numbers, plus the outputs of the two ML models (the Linear Regression forecast
and the Isolation Forest anomalies), into 5-8 tips. Each tip has a concrete
rupee amount. The rules themselves are written by hand - this part is not a
trained model. The tips are personal because every number in them comes from
the user's own transactions.
"""

import re

import numpy as np
import pandas as pd

from ai.categorizer import INCOME_CATEGORIES, clean_text, extract_merchant
from ai.forecast import forecast_next_month, monthly_totals

NEEDS = ["Rent", "Groceries", "Bills & Utilities", "Transport", "Health", "Education"]
WANTS = ["Food & Dining", "Shopping", "Entertainment", "Transfers"]  # any other category also counts as a want
REQUIRED_COLUMNS = ["date", "description", "amount", "type"]
MIN_TIPS, MAX_TIPS = 5, 8


# ---------------------------------------------------------------------------
# Part 1: analysis helpers
# ---------------------------------------------------------------------------
def format_inr(amount):
    """Format a number the Indian way: 123456 -> '₹1,23,456'."""
    digits = str(int(round(abs(amount))))
    if len(digits) > 3:
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", digits[:-3])
        digits = f"{head},{digits[-3:]}"
    return ("-₹" if amount < 0 else "₹") + digits


def parse_dates(values):
    """Read dates written as 2026-05-01 (ISO) or, failing that, 01/05/2026 (Indian day/month/year)."""
    dates = pd.to_datetime(values, format="ISO8601", errors="coerce")
    failed = dates.isna() & values.notna()
    if failed.any():
        dates[failed] = pd.to_datetime(values[failed], format="mixed", dayfirst=True, errors="coerce")
    return dates


def prepare_transactions(df):
    """Clean a transactions table (sample data or a user's CSV).

    Needs columns date, description, amount, type; category is optional.
    Amounts become positive numbers, type becomes 'income' or 'expense',
    and empty categories become missing (so the AI can fill them in).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing column(s): {', '.join(missing)}")
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = np.nan
    df["date"] = parse_dates(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "description"])
    df["amount"] = df["amount"].abs().astype(float)
    df["description"] = df["description"].astype(str).str.strip()
    kind = df["type"].astype(str).str.strip().str.lower()
    kind = kind.replace({"credit": "income", "cr": "income", "debit": "expense", "dr": "expense"})
    df["type"] = kind.where(kind.isin(["income", "expense"]), "expense")
    df["category"] = df["category"].map(lambda c: c.strip() if isinstance(c, str) and c.strip() else np.nan)
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


def month_count(df):
    """Number of calendar months the data covers (at least 1)."""
    months = df["date"].dt.to_period("M")
    return max(1, len(pd.period_range(months.min(), months.max(), freq="M")))


def expenses_only(df):
    """Just the expense rows."""
    return df[df["type"] == "expense"]


def overall_totals(df):
    """Total income, expenses, savings and savings rate (%) for the whole period."""
    income = df.loc[df["type"] == "income", "amount"].sum()
    expense = expenses_only(df)["amount"].sum()
    savings = income - expense
    rate = savings / income * 100 if income > 0 else 0.0
    return {"income": income, "expense": expense, "savings": savings, "savings_rate": rate}


def monthly_summary(df):
    """Income, expense, savings and savings rate for every month."""
    months = df["date"].dt.to_period("M").rename("month")
    table = df.pivot_table(index=months, columns="type", values="amount", aggfunc="sum", fill_value=0)
    table = table.reindex(columns=["income", "expense"], fill_value=0)
    table["savings"] = table["income"] - table["expense"]
    table["savings_rate"] = np.where(table["income"] > 0, table["savings"] / table["income"] * 100, 0.0)
    table.index = table.index.strftime("%b %Y")
    return table.reset_index()


def category_totals(df):
    """Total expense per category, largest first."""
    return (expenses_only(df).groupby("category")["amount"].sum()
            .sort_values(ascending=False).reset_index())


def month_over_month(df):
    """Change in each category between the last two months of data."""
    table = monthly_totals(df)
    if len(table) < 2:
        return pd.DataFrame()
    previous, latest = table.iloc[-2], table.iloc[-1]
    change = latest - previous
    result = pd.DataFrame({
        "category": table.columns,
        "previous": previous.to_numpy(),
        "latest": latest.to_numpy(),
        "change": change.to_numpy(),
        "change_pct": np.where(previous > 0, change / previous.replace(0, np.nan) * 100, np.nan),
    })
    result.attrs["months"] = (table.index[-2].strftime("%b %Y"), table.index[-1].strftime("%b %Y"))
    return result.sort_values("change", ascending=False).reset_index(drop=True)


def top_merchants(df, n=10, categories=None):
    """Merchants you spend the most with (optionally only within some categories)."""
    exp = expenses_only(df)
    if categories is not None:
        exp = exp[exp["category"].isin(categories)]
    if exp.empty:
        return pd.DataFrame(columns=["merchant", "category", "payments", "total", "average"])
    exp = exp.assign(merchant=exp["description"].map(extract_merchant))
    exp = exp.assign(key=exp["merchant"].str.lower())
    grouped = exp.groupby("key").agg(
        merchant=("merchant", lambda s: s.mode().iloc[0]),
        category=("category", lambda s: s.mode().iloc[0]),
        payments=("amount", "size"),
        total=("amount", "sum"),
        average=("amount", "mean"),
    )
    return grouped.sort_values("total", ascending=False).head(n).reset_index(drop=True)


def weekday_weekend_split(df, categories=None):
    """Average spend per weekday vs per weekend day, for each category."""
    exp = expenses_only(df)
    if categories is not None:
        exp = exp[exp["category"].isin(categories)]
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    weekend_days = int((days.day_of_week >= 5).sum())
    weekdays = len(days) - weekend_days
    day_type = np.where(exp["date"].dt.day_of_week >= 5, "Weekend", "Weekday")
    table = exp.pivot_table(index="category", columns=day_type, values="amount", aggfunc="sum", fill_value=0)
    table = table.reindex(columns=["Weekday", "Weekend"], fill_value=0)
    table["Weekday"] = table["Weekday"] / max(weekdays, 1)
    table["Weekend"] = table["Weekend"] / max(weekend_days, 1)
    table.attrs["days"] = {"weekdays": weekdays, "weekend_days": weekend_days}
    return table.rename(columns={"Weekday": "per_weekday", "Weekend": "per_weekend_day"}).reset_index()


def bucket_of(category):
    """Which 50/30/20 bucket an expense category belongs to."""
    return "Needs" if category in NEEDS else "Wants"


def needs_wants_savings(df):
    """Average monthly income, needs, wants and savings, plus each as a % of income."""
    months = month_count(df)
    income = df.loc[df["type"] == "income", "amount"].sum() / months
    exp = expenses_only(df)
    by_category = exp.groupby("category")["amount"].sum() / months
    needs = by_category[[bucket_of(c) == "Needs" for c in by_category.index]].sum()
    wants = by_category[[bucket_of(c) == "Wants" for c in by_category.index]].sum()
    savings = income - needs - wants
    pct = (lambda x: x / income * 100) if income > 0 else (lambda x: 0.0)
    return {"months": months, "income": income, "needs": needs, "wants": wants, "savings": savings,
            "needs_pct": pct(needs), "wants_pct": pct(wants), "savings_pct": pct(savings),
            "by_category": by_category.sort_values(ascending=False)}


def detect_subscriptions(df):
    """Find recurring payments: the same description and (almost) the same amount
    (within 5%) in at least 3 different months, about once a month or less often.

    monthly_cost = total paid / months between the first and last payment,
    so something paid every second month counts at half its price per month.
    """
    columns = ["name", "category", "amount", "monthly_cost", "payments", "last_paid"]
    exp = expenses_only(df)
    exp = exp[~exp["category"].isin(["Rent", "Transfers"] + INCOME_CATEGORIES)]
    if exp.empty:
        return pd.DataFrame(columns=columns)
    exp = exp.assign(key=exp["description"].map(clean_text), month=exp["date"].dt.to_period("M"))
    rows = []
    for _, group in exp.groupby("key"):
        months_seen = group["month"].nunique()
        mean, spread = group["amount"].mean(), group["amount"].std(ddof=0)
        if months_seen < 3 or len(group) > months_seen + 1 or spread > 0.05 * mean:
            continue
        span = (group["month"].max() - group["month"].min()).n + 1
        rows.append({"name": extract_merchant(group["description"].iloc[0]),
                     "category": group["category"].mode().iloc[0], "amount": mean,
                     "monthly_cost": group["amount"].sum() / span, "payments": len(group),
                     "last_paid": group["date"].max().date()})
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("monthly_cost", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 2: recommendations. Every tip is a dict: title, detail, monthly_saving.
# ---------------------------------------------------------------------------
def tip(title, detail, monthly_saving=0.0):
    """Build one recommendation."""
    return {"title": title, "detail": detail, "monthly_saving": max(0.0, float(monthly_saving))}


def savings_rate_tip(nws):
    """Compare the user's savings rate with the 20% target."""
    income, savings, rate = nws["income"], nws["savings"], nws["savings_pct"]
    if income <= 0:
        return None
    if rate >= 20:
        return tip(f"Your savings rate of {rate:.1f}% beats the 20% target",
                   f"You keep {format_inr(savings)} of your {format_inr(income)} monthly income. "
                   f"Moving that {format_inr(savings)} to a separate savings account on salary day "
                   f"keeps it from being spent by accident.")
    gap = 0.20 * income - savings
    start = (f"you keep {format_inr(savings)} a month" if savings >= 0
             else f"you spend {format_inr(-savings)} a month more than you earn")
    return tip(f"Raise your savings rate from {rate:.1f}% to 20%",
               f"On average you earn {format_inr(income)} and {start}. Saving {format_inr(gap)} more "
               f"each month reaches the 20% target - {format_inr(gap * 12)} extra in a year.", gap)


def fifty_thirty_twenty_tip(nws):
    """The 50/30/20 rule: 50% needs, 30% wants, 20% savings."""
    if nws["income"] <= 0:
        return None
    parts = []
    if nws["needs_pct"] > 50:
        rent_pct = nws["by_category"].get("Rent", 0) / nws["income"] * 100
        parts.append(f"Needs take {nws['needs_pct']:.1f}% of income, {nws['needs_pct'] - 50:.1f} points over "
                     f"the 50% guide (rent alone is {rent_pct:.0f}%).")
    else:
        parts.append(f"Needs are {nws['needs_pct']:.0f}% of income - inside the 50% guide.")
    excess = nws["wants"] - 0.30 * nws["income"]
    wants = nws["by_category"][[bucket_of(c) == "Wants" for c in nws["by_category"].index]]
    biggest = ", ".join(f"{c} {format_inr(v)}" for c, v in wants.head(3).items())
    if excess > 0:
        parts.append(f"Wants are {nws['wants_pct']:.0f}% (biggest: {biggest} a month); trimming "
                     f"{format_inr(excess)} a month brings them down to 30%.")
    else:
        parts.append(f"Wants are {nws['wants_pct']:.0f}% - inside the 30% guide (biggest: {biggest}).")
    title = (f"50/30/20 check: needs {nws['needs_pct']:.0f}% · wants {nws['wants_pct']:.0f}% · "
             f"savings {nws['savings_pct']:.0f}%")
    return tip(title, " ".join(parts), excess)


def trending_tips(forecast, table, next_month, limit=2):
    """Categories that are climbing: the regression line slopes up, the forecast is
    at least 10% above the average, and the latest month is already above average."""
    rising = forecast[(forecast["trend_per_month"] > 0) & (forecast["average"] >= 500)
                      & (forecast["forecast"] > forecast["average"] * 1.10)
                      & (forecast["last_month"] > forecast["average"])]
    rising = rising.assign(extra=rising["forecast"] - rising["average"]).sort_values("extra", ascending=False)
    first, last = table.index[0].strftime("%b"), table.index[-1].strftime("%b")
    tips = []
    for row in rising.head(limit).itertuples():
        start, end = table[row.category].iloc[0], table[row.category].iloc[-1]
        tips.append(tip(
            f"{row.category} is trending up - forecast {format_inr(row.forecast)} for {next_month}",
            f"It went from {format_inr(start)} in {first} to {format_inr(end)} in {last}, rising about "
            f"{format_inr(row.trend_per_month)} a month. Holding it at your {len(table)}-month average of "
            f"{format_inr(row.average)} would save {format_inr(row.extra)} next month.", row.extra))
    return tips


def subscription_tip(subscriptions):
    """Streaming and app subscriptions found by detect_subscriptions()."""
    streaming = subscriptions[subscriptions["category"] == "Entertainment"]
    if streaming.empty:
        return None
    total = streaming["monthly_cost"].sum()
    listing = ", ".join(f"{r.name} {format_inr(r.amount)}" for r in streaming.itertuples())
    if len(streaming) == 1:
        return tip(f"Your {streaming['name'].iloc[0]} subscription costs {format_inr(total * 12)} a year",
                   f"Recurring payment found: {listing}. Check you still use it every month.")
    keep = streaming.iloc[0]  # the most expensive one
    saving = total - keep["monthly_cost"]
    return tip(f"{len(streaming)} streaming/app subscriptions cost {format_inr(total)} a month",
               f"Recurring payments found: {listing} - {format_inr(total * 12)} a year. Keeping only "
               f"{keep['name']} and rotating the others one month at a time saves {format_inr(saving)} a month.",
               saving)


def top_merchant_tip(df, months):
    """The merchant the user spends most with on wants, as a habit (at least 3 payments)."""
    top = top_merchants(df, n=50, categories=[c for c in WANTS if c != "Transfers"])
    top = top[top["payments"] >= 3]
    if top.empty:
        return None
    row = top.iloc[0]
    per_month = row["total"] / months
    return tip(f"{row['merchant']} is your biggest discretionary spend: {format_inr(row['total'])} "
               f"in {months} months",
               f"{row['payments']} payments of about {format_inr(row['average'])} each - "
               f"{format_inr(per_month)} a month. Cutting a quarter of them saves about "
               f"{format_inr(per_month / 4)} a month.", per_month / 4)


def weekend_tip(df, months):
    """Wants spending on weekend days compared with weekdays."""
    split = weekday_weekend_split(df, categories=[c for c in WANTS if c != "Transfers"])
    weekday, weekend = split["per_weekday"].sum(), split["per_weekend_day"].sum()
    if weekday <= 0 or weekend < 1.25 * weekday:
        return None
    weekend_days_per_month = split.attrs["days"]["weekend_days"] / months
    saving = (weekend - 1.25 * weekday) * weekend_days_per_month
    return tip(f"Weekends cost you {weekend / weekday:.1f}× more than weekdays",
               f"On food, shopping and entertainment you spend {format_inr(weekend)} per weekend day vs "
               f"{format_inr(weekday)} per weekday. A weekend budget of {format_inr(1.25 * weekday)} a day "
               f"saves about {format_inr(saving)} a month.", saving)


def anomaly_tip(anomalies, months):
    """One-off spikes found by the Isolation Forest."""
    if anomalies is None or not anomalies["is_anomaly"].any():
        return None
    flagged = anomalies[anomalies["is_anomaly"]]
    biggest = flagged.loc[flagged["amount"].idxmax()]
    total = flagged["amount"].sum()
    return tip(f"{len(flagged)} unusual expenses added up to {format_inr(total)}",
               f"The biggest was \"{biggest['description']}\" ({format_inr(biggest['amount'])} on "
               f"{biggest['date']:%d %b}). Putting {format_inr(total / months)} a month into a separate "
               f"one-off fund would cover spikes like these without touching your savings.")


def emergency_fund_tip(nws):
    """Six months of essential costs as a safety net."""
    target = 6 * nws["needs"]
    if nws["savings"] > 0:
        timing = (f"At your current savings of {format_inr(nws['savings'])} a month that takes about "
                  f"{target / nws['savings']:.0f} months.")
    else:
        timing = "Start by getting your monthly savings above zero."
    return tip(f"Build a 6-month emergency fund of {format_inr(target)}",
               f"Your essential costs (needs) are {format_inr(nws['needs'])} a month. {timing}")


def generate_recommendations(df, anomalies=None):
    """Build 5-8 personalised tips, the ones that save the most money first.

    `anomalies` is the output of ai.anomaly.detect_anomalies(). One-off spikes
    are left out of the habit-based tips (trend, top merchant, weekends) so a
    single big purchase does not distort them.
    """
    nws = needs_wants_savings(df)
    months = nws["months"]
    exclude = anomalies.index[anomalies["is_anomaly"]] if anomalies is not None else None
    regular = df.drop(index=exclude) if exclude is not None else df
    forecast, table, next_month = forecast_next_month(df, exclude_index=exclude)

    tips = [savings_rate_tip(nws), fifty_thirty_twenty_tip(nws)]
    tips += trending_tips(forecast, table, next_month)
    tips += [subscription_tip(detect_subscriptions(df)), top_merchant_tip(regular, months),
             weekend_tip(regular, months), anomaly_tip(anomalies, months)]
    tips = [t for t in tips if t is not None]
    tips.sort(key=lambda t: t["monthly_saving"], reverse=True)
    if len(tips) < MAX_TIPS:
        tips.append(emergency_fund_tip(nws))
    return tips[:MAX_TIPS]
