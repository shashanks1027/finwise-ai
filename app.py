"""
FinWise AI - an AI-powered personal finance manager (Streamlit app).

Run from the finwise-ai folder:
    streamlit run app.py

The sample data (data/transactions.csv) loads automatically, so every page
works straight away. Use the sidebar to move between pages.

Where the AI lives (all in the ai/ folder):
  * categorizer.py - TF-IDF + Logistic Regression: category from the description
  * anomaly.py     - Isolation Forest: flags unusual expenses
  * forecast.py    - Linear Regression: next month's spend per category
  * recommender.py - spending analysis + rule-based savings tips built on the models above
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai.anomaly import CONTAMINATION, detect_anomalies
from ai.categorizer import accuracy_on, categorize_missing, top_predictions, train_categorizer
from ai.forecast import backtest, forecast_next_month, monthly_totals
from ai.recommender import (INCOME_CATEGORIES, NEEDS, WANTS, category_totals, detect_subscriptions,
                            format_inr, generate_recommendations, month_over_month, monthly_summary,
                            needs_wants_savings, overall_totals, prepare_transactions, top_merchants,
                            weekday_weekend_split)

SAMPLE_CSV = Path(__file__).resolve().parent / "data" / "transactions.csv"
COLUMNS = ["date", "description", "amount", "type", "category", "source", "confidence"]
COLORS = {"income": "#2E9E5B", "expense": "#D9534F", "savings": "#3B7DD8", "forecast": "#F0A202"}
PAGES = ["Dashboard", "Add Transaction", "Upload CSV", "Spending Patterns",
         "Anomalies", "Forecast", "Recommendations", "AI Model"]

# How tables show each column.
TABLE_COLUMNS = {
    "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
    "description": st.column_config.TextColumn("Description", width="large"),
    "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
    "type": "Type",
    "category": "Category",
    "source": "Source",
    "confidence": st.column_config.ProgressColumn("AI confidence", min_value=0, max_value=1, format="percent"),
}

# A small file in the upload format (category left blank) so the upload can be tried in a demo.
EXAMPLE_UPLOAD = """date,description,amount,type,category
2026-10-26,Chai Point - Masala chai and samosa,140,expense,
2026-10-26,UPI/Rapido/rapido@ybl,86,expense,
2026-10-27,Zepto - Daily essentials,386,expense,
2026-10-27,Airtel Postpaid - Bill payment,599,expense,
2026-10-28,PVR Cinemas - 2 tickets,720,expense,
2026-10-28,Tata 1mg - Vitamin tablets,410,expense,
2026-10-29,Myntra - Order #77120,1499,expense,
2026-10-29,Fiverr - Logo design payout,4200,income,
2026-10-30,Sent to Kavya Menon,500,expense,
2026-10-30,IRCTC - Train ticket to Mysuru,345,expense,
2026-10-31,Coursera - Machine learning course,2499,expense,
2026-10-31,Domino's Pizza - Takeaway,549,expense,
"""

st.set_page_config(page_title="FinWise AI", page_icon="💰", layout="wide")


# ---------------------------------------------------------------------------
# Data and models (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    """The demo transactions, cleaned and ready to use."""
    df = prepare_transactions(pd.read_csv(SAMPLE_CSV))
    return df.assign(source="Sample data", confidence=np.nan)[COLUMNS]


@st.cache_resource(show_spinner="Training the AI categorizer...")
def get_categorizer():
    """Train the categorizer once; every page and every rerun reuses it."""
    result = train_categorizer()
    result["demo_accuracy"] = accuracy_on(result["model"], load_sample_data())
    return result


@st.cache_data
def find_anomalies(df):
    """Isolation Forest results for the current data (re-run only when the data changes)."""
    return detect_anomalies(df)


def get_transactions():
    """The transactions for this browser session (starts as the sample data)."""
    if "transactions" not in st.session_state:
        st.session_state.transactions = load_sample_data()
    return st.session_state.transactions


def add_transactions(new_rows):
    """Append rows to this session's transactions, keeping them in date order."""
    combined = pd.concat([get_transactions(), new_rows[COLUMNS]], ignore_index=True)
    st.session_state.transactions = combined.sort_values("date", kind="mergesort").reset_index(drop=True)


def anomaly_index(df):
    """Row labels of the expenses the Isolation Forest flagged."""
    scored = find_anomalies(df)
    return scored.index[scored["is_anomaly"]]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard(df):
    """Feature 1: totals, monthly income vs expenses, category breakdown."""
    st.title("💰 Dashboard")
    st.caption(f"{len(df)} transactions from {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")
    totals = overall_totals(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total income", format_inr(totals["income"]), border=True)
    c2.metric("Total expenses", format_inr(totals["expense"]), border=True)
    c3.metric("Savings", format_inr(totals["savings"]), border=True)
    c4.metric("Savings rate", f"{totals['savings_rate']:.1f}%", border=True,
              delta=f"{totals['savings_rate'] - 20:+.1f} pts vs 20% target")

    left, right = st.columns([3, 2])
    monthly = monthly_summary(df)
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["income"], name="Income", marker_color=COLORS["income"])
    fig.add_bar(x=monthly["month"], y=monthly["expense"], name="Expenses", marker_color=COLORS["expense"])
    fig.add_scatter(x=monthly["month"], y=monthly["savings"], name="Savings", mode="lines+markers",
                    line_color=COLORS["savings"])
    fig.update_layout(barmode="group", yaxis_title="₹", legend=dict(orientation="h"), margin=dict(t=10))
    left.subheader("Monthly income vs expenses")
    left.plotly_chart(fig)

    pie = px.pie(category_totals(df), names="category", values="amount", hole=0.45)
    pie.update_traces(textposition="inside", textinfo="percent+label")
    pie.update_layout(showlegend=False, margin=dict(t=10))
    right.subheader("Where the money goes")
    right.plotly_chart(pie)

    st.subheader("All transactions")
    st.dataframe(df.sort_values("date", ascending=False), hide_index=True, column_config=TABLE_COLUMNS)


def save_new_transaction(suggested, confidence, category_key, type_key):
    """Button callback for the Add Transaction page: check the inputs and add the row."""
    state = st.session_state
    description = state.get("new_description", "").strip()
    if not description or state.new_amount <= 0:
        state.add_message = ("error", "Please enter a description and an amount above ₹0.")
        return
    category = state[category_key]
    used_ai = category == suggested
    row = pd.DataFrame([{
        "date": pd.Timestamp(state.new_date), "description": description, "amount": float(state.new_amount),
        "type": state[type_key], "category": category,
        "source": "Added - AI category" if used_ai else "Added - category changed by you",
        "confidence": confidence if used_ai else np.nan,
    }])
    add_transactions(row)
    state.add_message = ("success", f"Added \"{description}\" ({format_inr(state.new_amount)}) as {category}.")
    state.new_description = ""


def page_add_transaction(df, categorizer):
    """Feature 3: the category is predicted live while the user types; the user can override it."""
    st.title("➕ Add a transaction")
    st.caption("Start typing a description - the AI suggests a category as you type.")
    if "add_message" in st.session_state:
        kind, text = st.session_state.pop("add_message")
        (st.success if kind == "success" else st.error)(text)

    model = categorizer["model"]
    categories = sorted(model.classes_)
    description = st.text_input("Description", key="new_description", live=True,
                                placeholder="e.g. Swiggy order, Uber to office, Netflix subscription")
    suggested, confidence = None, np.nan
    if description.strip():
        top3 = top_predictions(model, description)
        suggested, confidence = top3[0]
        st.info(f"🤖 AI suggestion: **{suggested}** - {confidence:.0%} confident")
        chart = pd.DataFrame(top3, columns=["category", "probability"])
        fig = px.bar(chart, x="probability", y="category", orientation="h", text_auto=".0%", range_x=[0, 1])
        fig.update_layout(height=180, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed", title=""),
                          xaxis=dict(title="Probability for the top 3 categories", tickformat=".0%"))
        st.plotly_chart(fig)
        if confidence < 0.5:
            st.warning("The model is not sure about this one - please check the category below.")

    col1, col2, col3 = st.columns(3)
    col1.date_input("Date", value=df["date"].max().date(), key="new_date")
    col2.number_input("Amount (₹)", min_value=0.0, step=10.0, key="new_amount")
    default_type = "income" if suggested in INCOME_CATEGORIES else "expense"
    type_key = f"new_type_{default_type}"  # a new key when the suggestion changes resets the default
    col3.selectbox("Type", ["expense", "income"], index=["expense", "income"].index(default_type), key=type_key)
    category_key = f"new_category_{suggested}"
    st.selectbox("Category (change it if the AI got it wrong)", categories, key=category_key,
                 index=categories.index(suggested) if suggested else 0)
    st.button("Add transaction", type="primary", on_click=save_new_transaction,
              args=(suggested, confidence, category_key, type_key), disabled=not description.strip())

    added = df[df["source"].str.startswith("Added")]
    if not added.empty:
        st.subheader("Added in this session")
        st.dataframe(added, hide_index=True, column_config=TABLE_COLUMNS)


def page_upload(categorizer):
    """Feature 4: upload a CSV; rows without a category are categorised by the model."""
    st.title("📤 Upload your own CSV")
    st.markdown("Use the same columns as the sample data: **date, description, amount, type, category**.  \n"
                "`category` can be left empty - the AI fills it in. Dates may be `2026-10-31` or `31/10/2026`; "
                "`type` is `income` or `expense` (`credit` / `debit` also work).")
    st.download_button("Download an example CSV (categories left blank)", EXAMPLE_UPLOAD,
                       file_name="example_upload.csv", mime="text/csv")
    file = st.file_uploader("Choose a CSV file", type="csv")
    if file is None:
        return
    try:
        cleaned = prepare_transactions(pd.read_csv(file))
    except Exception as error:  # a broken file should show a message, not crash the app
        st.error(f"Could not read this file: {error}")
        return
    if cleaned.empty:
        st.error("No valid rows found - check the date and amount columns.")
        return

    result = categorize_missing(categorizer["model"], cleaned)
    n_ai = int((result["categorized_by"] == "AI").sum())
    st.success(f"Read {len(result)} rows. The AI categorised the {n_ai} rows that had no category.")
    st.dataframe(result[["date", "description", "amount", "type", "category", "categorized_by", "confidence"]],
                 hide_index=True, column_config={**TABLE_COLUMNS, "categorized_by": "Categorised by"})

    used = st.session_state.setdefault("used_uploads", set())
    if file.file_id in used:
        st.info("This file has already been added. Open the other pages to see the results.")
        return
    mode = st.radio("What should happen to these rows?",
                    ["Add them to the current data", "Replace the current data with them"], horizontal=True)
    if st.button("Use this data", type="primary"):
        rows = result.assign(source=np.where(result["categorized_by"] == "AI", "Uploaded - AI category", "Uploaded"))
        if mode.startswith("Replace"):
            st.session_state.transactions = rows[COLUMNS].reset_index(drop=True)
        else:
            add_transactions(rows)
        used.add(file.file_id)
        st.rerun()


def page_patterns(df):
    """Feature 5: month-over-month change, top merchants, weekday vs weekend."""
    st.title("📊 Spending patterns")

    st.subheader("Month-over-month change by category")
    mom = month_over_month(df)
    if mom.empty:
        st.info("At least two months of data are needed.")
    else:
        previous, latest = mom.attrs["months"]
        st.caption(f"{previous} → {latest}")
        colors = np.where(mom["change"] > 0, COLORS["expense"], COLORS["income"])
        fig = go.Figure(go.Bar(x=mom["change"], y=mom["category"], orientation="h", marker_color=colors))
        fig.update_layout(xaxis_title=f"Change in ₹ ({latest} minus {previous})", height=380,
                          yaxis=dict(autorange="reversed"), margin=dict(t=10))
        left, right = st.columns([3, 2])
        left.plotly_chart(fig)
        right.dataframe(mom, hide_index=True, column_config={
            "category": "Category", "previous": st.column_config.NumberColumn(previous, format="₹%.0f"),
            "latest": st.column_config.NumberColumn(latest, format="₹%.0f"),
            "change": st.column_config.NumberColumn("Change", format="₹%+.0f"),
            "change_pct": st.column_config.NumberColumn("Change %", format="%+.1f%%")})

    table = monthly_totals(df)
    heat = px.imshow(table.T, x=[m.strftime("%b %Y") for m in table.index], text_auto=".3s", aspect="auto",
                     color_continuous_scale="Reds", labels=dict(color="₹"))
    heat.update_layout(margin=dict(t=10), xaxis_title="", yaxis_title="")
    st.subheader("Monthly spend by category")
    st.plotly_chart(heat)

    st.subheader("Top merchants")
    merchants = top_merchants(df, n=10)
    left, right = st.columns([3, 2])
    fig = px.bar(merchants, x="total", y="merchant", color="category", orientation="h")
    fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis_title="Total spent (₹)", margin=dict(t=10))
    left.plotly_chart(fig)
    right.dataframe(merchants, hide_index=True, column_config={
        "merchant": "Merchant", "category": "Category", "payments": "Payments",
        "total": st.column_config.NumberColumn("Total", format="₹%.0f"),
        "average": st.column_config.NumberColumn("Average", format="₹%.0f")})

    st.subheader("Weekday vs weekend")
    split = weekday_weekend_split(df)
    weekday, weekend = split["per_weekday"].sum(), split["per_weekend_day"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average spend per weekday", format_inr(weekday), border=True)
    c2.metric("Average spend per weekend day", format_inr(weekend), border=True)
    c3.metric("Weekend vs weekday", f"{weekend / weekday:.1f}×" if weekday else "-", border=True)
    long = split.melt(id_vars="category", value_vars=["per_weekday", "per_weekend_day"],
                      var_name="day", value_name="per_day")
    long["day"] = long["day"].map({"per_weekday": "Per weekday", "per_weekend_day": "Per weekend day"})
    fig = px.bar(long, x="category", y="per_day", color="day", barmode="group",
                 labels={"per_day": "Average ₹ per day", "category": "", "day": ""})
    fig.update_layout(legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig)


def page_anomalies(df):
    """Feature 6: unusual expenses, each with a plain-English reason."""
    st.title("🚨 Unusual expenses")
    st.caption("Found by an Isolation Forest that compares each expense with what is normal for you "
               "in that category and at that merchant.")
    scored = find_anomalies(df)
    flagged = scored[scored["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expenses checked", len(scored), border=True)
    c2.metric("Flagged as unusual", len(flagged), border=True)
    c3.metric("Value of flagged expenses", format_inr(flagged["amount"].sum()), border=True)
    if flagged.empty:
        st.success("Nothing unusual found.")
    for row in flagged.itertuples():
        with st.container(border=True):
            st.markdown(f"**{row.description}** · {format_inr(row.amount)} · {row.date:%d %b %Y} · {row.category}")
            st.markdown(f"⚠️ {row.reason}")

    if not scored.empty:
        st.subheader("Every expense (log scale)")
        plot = scored.assign(status=np.where(scored["is_anomaly"], "Unusual", "Normal"))
        fig = px.scatter(plot, x="date", y="amount", color="status", log_y=True,
                         hover_data=["description", "category"],
                         color_discrete_map={"Normal": "#9AA5B1", "Unusual": COLORS["expense"]})
        fig.update_layout(yaxis_title="Amount (₹, log scale)", xaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig)


def page_forecast(df):
    """Feature 7: next month's spend per category, next to the actuals."""
    exclude = anomaly_index(df)
    forecast, table, next_month = forecast_next_month(df, exclude_index=exclude)
    st.title(f"🔮 Forecast for {next_month}")
    st.caption("A Linear Regression line is fitted to each category's monthly totals and extended one month "
               "ahead. One-off anomalies are left out so they do not bend the line.")
    last_label = table.index[-1].strftime("%b %Y")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast total, {next_month}", format_inr(forecast["forecast"].sum()), border=True)
    c2.metric(f"Actual, {last_label}", format_inr(forecast["last_month"].sum()), border=True)
    c3.metric(f"Monthly average ({len(table)} months)", format_inr(forecast["average"].sum()), border=True)

    fig = go.Figure()
    fig.add_bar(x=forecast["category"], y=forecast["last_month"], name=f"Actual {last_label}",
                marker_color=COLORS["savings"])
    fig.add_bar(x=forecast["category"], y=forecast["average"], name="Actual monthly average", marker_color="#9AA5B1")
    fig.add_bar(x=forecast["category"], y=forecast["forecast"], name=f"Forecast {next_month}",
                marker_color=COLORS["forecast"])
    fig.update_layout(barmode="group", yaxis_title="₹", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig)
    st.dataframe(forecast, hide_index=True, column_config={
        "category": "Category",
        "last_month": st.column_config.NumberColumn(f"Actual {last_label}", format="₹%.0f"),
        "average": st.column_config.NumberColumn("Monthly average", format="₹%.0f"),
        "forecast": st.column_config.NumberColumn(f"Forecast {next_month}", format="₹%.0f"),
        "trend_per_month": st.column_config.NumberColumn("Trend per month", format="₹%+.0f")})

    st.subheader("One category in detail")
    category = st.selectbox("Category", forecast["category"])
    row = forecast.set_index("category").loc[category]
    months = [m.strftime("%b %Y") for m in table.index]
    x = np.arange(len(months))
    trend_line = row["average"] + row["trend_per_month"] * (x - x.mean())  # the fitted regression line
    fig = go.Figure()
    fig.add_scatter(x=months, y=table[category], name="Actual", mode="lines+markers", line_color=COLORS["savings"])
    fig.add_scatter(x=months, y=trend_line, name="Regression line", mode="lines", line=dict(dash="dot", color="#9AA5B1"))
    fig.add_scatter(x=[months[-1], next_month], y=[table[category].iloc[-1], row["forecast"]], name="Forecast",
                    mode="lines+markers", line=dict(dash="dash", color=COLORS["forecast"]))
    fig.update_layout(yaxis_title="₹", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig)

    st.subheader("How accurate is the forecast?")
    test = backtest(df, exclude_index=exclude)
    if test is None:
        st.info("At least three months of data are needed to test the forecast.")
        return
    lr_error, base_error = test["lr_error_pct"].mean(), test["baseline_error_pct"].mean()
    st.markdown(f"For each of the last {len(test)} months we trained only on the months before it and "
                f"predicted that month's total. Mean error: **Linear Regression {lr_error:.1f}%**, "
                f"plain average of earlier months {base_error:.1f}%.")
    st.dataframe(test, hide_index=True, column_config={
        "month": "Month hidden from the model",
        "actual": st.column_config.NumberColumn("Actual", format="₹%.0f"),
        "linear_regression": st.column_config.NumberColumn("Linear Regression", format="₹%.0f"),
        "average_baseline": st.column_config.NumberColumn("Average baseline", format="₹%.0f"),
        "lr_error_pct": st.column_config.NumberColumn("LR error", format="%.1f%%"),
        "baseline_error_pct": st.column_config.NumberColumn("Baseline error", format="%.1f%%")})
    if base_error < lr_error:
        st.caption("With only a few months of history a straight line reacts strongly to one noisy month, so a "
                   "plain average is currently more accurate on totals. The regression's advantage is that it "
                   "shows the direction of each category (the trend column); more months of data make it steadier.")


def page_recommendations(df):
    """Feature 8: 5-8 personalised tips with rupee amounts."""
    st.title("💡 Personalised recommendations")
    st.caption("Every tip is calculated from your own transactions, the AI forecast and the anomaly detector. "
               "Biggest possible saving first.")
    tips = generate_recommendations(df, find_anomalies(df))
    for i, tip in enumerate(tips, 1):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{i}. {tip['title']}**")
            left.write(tip["detail"])
            if tip["monthly_saving"] > 0:
                right.metric("Could save", f"{format_inr(tip['monthly_saving'])}/mo")

    st.subheader("50/30/20 rule")
    nws = needs_wants_savings(df)
    rule = pd.DataFrame({"bucket": ["Needs", "Wants", "Savings"] * 2,
                         "who": ["You"] * 3 + ["Target"] * 3,
                         "percent": [nws["needs_pct"], nws["wants_pct"], nws["savings_pct"], 50, 30, 20]})
    fig = px.bar(rule, x="bucket", y="percent", color="who", barmode="group", text_auto=".0f",
                 color_discrete_map={"You": COLORS["savings"], "Target": "#9AA5B1"},
                 labels={"percent": "% of income", "bucket": "", "who": ""})
    fig.update_layout(legend=dict(orientation="h"), margin=dict(t=10))
    left, right = st.columns([3, 2])
    left.plotly_chart(fig)
    right.markdown(f"**Needs:** {', '.join(NEEDS)}  \n**Wants:** {', '.join(WANTS)}  \n"
                   f"Average monthly income: **{format_inr(nws['income'])}**")

    st.subheader("Recurring payments detected")
    st.dataframe(detect_subscriptions(df), hide_index=True, column_config={
        "name": "Name", "category": "Category", "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
        "monthly_cost": st.column_config.NumberColumn("Cost per month", format="₹%.0f"),
        "payments": "Payments", "last_paid": st.column_config.DateColumn("Last paid", format="DD MMM YYYY")})


def page_model(df, categorizer):
    """Feature 2: how good the categorizer is - test accuracy and confusion matrix."""
    st.title("🧠 AI model: transaction categorizer")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy (held-out 20%)", f"{categorizer['accuracy']:.1%}", border=True)
    c2.metric("Accuracy on the demo transactions", f"{categorizer['demo_accuracy']:.1%}", border=True)
    c3.metric("Training / test examples", f"{categorizer['n_train']} / {categorizer['n_test']}", border=True)
    c4.metric("Categories", len(categorizer["labels"]), border=True)
    st.markdown(
        "**How it works.** Each description is cleaned (lower-case, numbers removed) and split into "
        "overlapping 2-4 letter pieces (*character n-grams*). **TF-IDF** turns these pieces into numbers, giving "
        "less weight to pieces that appear everywhere (like `upi`). **Logistic Regression** then learns how "
        "strongly each piece points to each category; its probability for the winning category is the "
        "confidence shown in the app. The test set is 20% of `training_data.csv` that the model never saw.")

    st.subheader("Confusion matrix (test set)")
    labels = categorizer["labels"]
    fig = px.imshow(categorizer["confusion"], x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted category", y="Actual category", color="Count"))
    fig.update_layout(height=560, margin=dict(t=10))
    st.plotly_chart(fig)
    st.caption("Numbers on the diagonal are correct predictions; anything off the diagonal is a mistake.")

    left, right = st.columns(2)
    left.subheader("Per-category scores")
    left.dataframe(categorizer["report"], column_config={
        "precision": st.column_config.NumberColumn("Precision", format="%.2f"),
        "recall": st.column_config.NumberColumn("Recall", format="%.2f"),
        "f1-score": st.column_config.NumberColumn("F1", format="%.2f"),
        "support": st.column_config.NumberColumn("Test examples", format="%d")})
    right.subheader("Test mistakes")
    right.dataframe(categorizer["mistakes"], hide_index=True)

    st.subheader("The other two models")
    scored = find_anomalies(df)
    test = backtest(df, exclude_index=anomaly_index(df))
    forecast_note = (f"Tested on the last {len(test)} months: mean error {test['lr_error_pct'].mean():.1f}% "
                     f"(plain average: {test['baseline_error_pct'].mean():.1f}%)." if test is not None else "")
    st.markdown(
        f"- **Isolation Forest** (Anomalies page) - features: amount ÷ your usual amount in the category, and "
        f"amount ÷ your usual amount at the merchant. Contamination {CONTAMINATION:.0%}. On the current data it "
        f"checked {len(scored)} expenses and flagged {int(scored['is_anomaly'].sum())}.\n"
        f"- **Linear Regression** (Forecast page) - one line per category through the monthly totals, extended "
        f"one month ahead. {forecast_note}")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def sidebar():
    """Page menu, a summary of the loaded data and a reset button."""
    st.sidebar.title("💰 FinWise AI")
    st.sidebar.caption("AI-powered personal finance manager")
    page = st.sidebar.radio("Go to", PAGES)
    df = get_transactions()
    st.sidebar.divider()
    st.sidebar.markdown(f"**{len(df)} transactions**  \n{df['date'].min():%d %b %Y} – {df['date'].max():%d %b %Y}")
    if st.sidebar.button("Reset to sample data"):
        for key in ["transactions", "used_uploads"]:
            st.session_state.pop(key, None)
        st.rerun()
    return page


def main():
    page = sidebar()
    df = get_transactions()
    categorizer = get_categorizer()
    if page == "Dashboard":
        page_dashboard(df)
    elif page == "Add Transaction":
        page_add_transaction(df, categorizer)
    elif page == "Upload CSV":
        page_upload(categorizer)
    elif page == "Spending Patterns":
        page_patterns(df)
    elif page == "Anomalies":
        page_anomalies(df)
    elif page == "Forecast":
        page_forecast(df)
    elif page == "Recommendations":
        page_recommendations(df)
    elif page == "AI Model":
        page_model(df, categorizer)


main()
