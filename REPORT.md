# FinWise AI: An AI-Powered Personal Finance Management System

**Project Report**

| | |
|---|---|
| Student name | [STUDENT NAME] |
| Roll no. | [ROLL NO] |
| College | [COLLEGE] |
| Course | [COURSE] |
| Subject | [SUBJECT] |

---

## 1. Abstract

FinWise AI is a working personal finance app built with Python and Streamlit. It tracks income and expenses, categorises transactions from their bank descriptions, detects unusual expenses, forecasts next month's spending and gives personalised savings tips. Three trained machine-learning models power it:

- a TF-IDF + Logistic Regression text classifier (96.5% accuracy on a held-out test set);
- an Isolation Forest anomaly detector (it caught all 7 planted unusual expenses);
- per-category Linear Regression forecasts.

A rule-based engine turns their outputs into 5–8 tips with rupee amounts. The app runs fully offline and ships with six months of realistic Indian sample data.

## 2. Problem Statement

Bank statements are long lists of cryptic lines such as `UPI/Swiggy/412345678`. Sorting them into categories by hand is tedious, so most people never do it. Without categories it is hard to notice that food delivery is creeping up, that four streaming subscriptions are running at once, or that one bill was unusually large. Generic advice like "spend less" does not help either; people need tips based on their own numbers.

## 3. Objectives

1. Show income, expenses, savings and savings rate at a glance.
2. Categorise transactions automatically, with a confidence score and the option to override.
3. Categorise rows in a user's own CSV.
4. Show patterns: month-over-month change, top merchants, weekday vs weekend spending.
5. Flag unusual expenses and explain why in plain words.
6. Forecast next month's spending per category.
7. Give 5–8 personalised savings tips with rupee amounts.
8. Stay reproducible (fixed seeds), offline and easy to explain.

## 4. How AI Tools Were Used to Develop the Project

The project was built with **Claude Code**, an AI coding assistant, working inside the project folder.

- **Design.** It proposed the module layout (one file per AI technique in `ai/`), chose each algorithm with a reason (Section 5) and designed a seeded synthetic dataset.
- **Code generation.** It wrote the generator, the four AI modules, the Streamlit app and the smoke test, with docstrings written for a student reader.
- **Debugging.** Running the code exposed real bugs, which were fixed:
  - ISO dates such as `2026-05-01` were misread as 5 January.
  - A CSV whose category column was completely empty crashed the categoriser.
  - An uploaded expense could be given an income category ("Salary"); predictions now respect the row's type.
  - Subscription detection allowed 15% price variation, so random cinema tickets counted as subscriptions; it now allows 5%.
  - One tip called a category "rising" while its latest month was lower than its first.
- **Testing.** `smoke_test.py` runs 26 automated checks:
  - the data is reproducible and accuracy is at least 85%;
  - every anomaly has a reason and every tip quotes a rupee amount;
  - CSV uploads are read and categorised.

  Every page was also run in a headless Streamlit test, and the server was started to confirm there were no tracebacks.
- **Honest evaluation.** Features and the forecast were compared with alternatives instead of reporting only favourable numbers.

The student reviewed the code, ran every command and is responsible for explaining it.

## 5. How AI Is Used Inside the App

### 5.1 Transaction categoriser: TF-IDF + Logistic Regression (`ai/categorizer.py`)

- **Input → output:** a description → one of 12 categories, with a confidence from 0 to 1. The top 3 categories update live as the user types.
- **How it works:**
  - Text is lower-cased and numbers are removed.
  - **TF-IDF** converts overlapping 2–4 letter pieces (character n-grams) into numbers, down-weighting pieces that appear everywhere, such as "upi".
  - **Logistic Regression** learns how strongly each piece points to each category, and its highest probability is the confidence.
  - For uploaded rows the model only chooses categories that match the type (an expense never becomes "Salary").
- **Why:** it is fast, works with little data, gives real probabilities and is easy to inspect. Character n-grams handle capitals and typos ("SWIGGY", "Swigy"). In 5-fold cross-validation they scored **96.9%**, against **91.7%** for word n-grams.
- **Results:**
  - **96.5%** accuracy on 144 held-out descriptions (576 used for training), with a macro F1 of **0.965**;
  - **95.6%** on the 435 demo transactions, which were never used for training;
  - 8 of 8 brand-new phrases correct.

### 5.2 Anomaly detector: Isolation Forest (`ai/anomaly.py`)

- **Input → output:** two features per expense → a flag, a score and a reason such as "12.8× what you usually pay at Uber (₹184)".
  - **Category feature:** amount ÷ the user's median in that category.
  - **Merchant feature:** amount ÷ the user's median at that merchant.
  - Both are log-scaled, and values below 0 are clipped, because spending less than usual is not a problem.
- **Why:** it needs no labelled examples. Random cuts isolate unusual points quickly. Because the features are ratios, rent (large but normal for rent) is never flagged. Contamination is set to 2%.
- **Results:** **9** of 421 expenses flagged, including **all 7 planted anomalies**, such as ₹18,990 headphones (13.4× typical Shopping). The other two flags are a fuel payment and a larger-than-usual transfer to a friend.

### 5.3 Forecaster: Linear Regression (`ai/forecast.py`)

- **Input → output:** month number (0, 1, 2, …) → that month's total for one category. The line is extended one month ahead, and its slope is the trend in ₹ per month. Anomalies are removed first so a one-off purchase does not bend the line.
- **Why:** with six monthly points, a straight line is the simplest model that still shows direction.
- **Results:** the November 2026 forecast is **₹58,746**. Food & Dining is rising by about ₹581 a month. In a rolling test, each of the last three months was hidden and then predicted. The mean error on total spend was **12.3%** for Linear Regression and **5.6%** for a plain average of earlier months. With little history the line overreacts to noisy months (Section 13).

### 5.4 Recommendation engine (`ai/recommender.py`)

This part is **rule-based, not a trained model**. It combines the user's own figures with the outputs of the two models above (forecast trends and anomalies), then ranks the tips by potential monthly saving:

- the 50/30/20 split and savings rate;
- top merchants and weekend spending;
- detected subscriptions.

## 6. Features

1. **Dashboard:** income, expenses, savings and savings rate; monthly income-vs-expense chart; category donut chart.
2. **AI Model page:** test accuracy, confusion matrix, per-category scores and test mistakes.
3. **Add Transaction:** live category prediction with confidence, a low-confidence warning and an override.
4. **CSV upload:** blank categories are filled in by the model; Indian dates and credit/debit are handled.
5. **Spending Patterns:** month-over-month change, monthly heatmap, top merchants, weekday vs weekend.
6. **Anomalies:** unusual expenses with plain-English reasons.
7. **Forecast:** next month per category next to the actuals, with a trend chart and an accuracy test.
8. **Recommendations:** 5–8 tips, a 50/30/20 chart and the recurring payments found.
9. **Sample data:** loads by default; **Reset to sample data** undoes changes.

## 7. System Architecture

```
                         data/generate_data.py  (seed 42)
                          |                       |
                          v                       v
               data/transactions.csv     data/training_data.csv
                          |                       |
                          v                       v
   user CSV / form --> prepare_transactions   train_categorizer
                          |                 (TF-IDF + LogisticRegression)
                          v                       |
                 session transactions <-----------+  fills missing categories
                          |
        +-----------------+------------------+---------------------+
        v                 v                  v                     v
   Dashboard &      ai/anomaly.py      ai/forecast.py       ai/recommender.py
   Patterns pages   IsolationForest -> LinearRegression ->  analysis + rules
   (pandas)         (flags spikes)     (spikes excluded)    -> 5-8 tips
        |                 |                  |                     |
        +-----------------+------------------+---------------------+
                                     v
                       app.py  (Streamlit UI, 8 pages)
```

The categoriser is trained once and cached with `st.cache_resource`. Anomaly detection is cached with `st.cache_data` and re-runs only when the data changes.

## 8. Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 (3.10+ supported) |
| User interface | Streamlit |
| Data handling | pandas, NumPy |
| Machine learning | scikit-learn (TfidfVectorizer, LogisticRegression, IsolationForest, LinearRegression) |
| Charts | Plotly |
| Development | Claude Code (AI coding assistant) |

There is no database, login or internet access.

## 9. Sample Dataset Description

`data/generate_data.py` writes both files with a fixed seed, and repeated runs produce byte-identical CSVs.

- **transactions.csv:** 435 rows from 1 May to 31 October 2026 (`date, description, amount, type, category`). The persona is a Bengaluru software engineer earning ₹78,000 a month.
  - **Income (14 rows):** salary, freelance, interest and cashback.
  - **Expenses (421 rows):** Food & Dining 125, Transport 92, Groceries 57, Entertainment 37, Shopping 31, Bills & Utilities 28, Health 22, Transfers 17, Rent 6 and Education 6.
  - **Descriptions:** Indian merchants (Swiggy, Zomato, Uber, Ola, BigBasket, Blinkit, Amazon, Flipkart, Netflix, Jio, BESCOM) in bank formats such as `UPI/Swiggy/…` and `POS/Flipkart/Bengaluru`.
  - **Built-in patterns:** rising food spending, heavier weekends, festive shopping in September–October, monthly subscriptions and 7 planted anomalies.
- **training_data.csv:** 720 labelled descriptions, 60 per category.
  - Mixed formats (UPI, POS, "Paid to…", NEFT/IMPS), upper and lower case, and typos.
  - Shared merchant names (Amazon / Amazon Fresh / Prime Video; Swiggy / Swiggy Instamart), so the model must learn from the other words.
  - Near-duplicates that differ only in numbers are removed.

## 10. Results

All figures come from `python smoke_test.py` (26/26 checks passed).

| Measure | Result |
|---|---|
| Total income / expenses (6 months) | ₹5,02,324 / ₹4,17,778 |
| Savings / savings rate | ₹84,546 / 16.8% |
| 50/30/20 split | needs 49.3%, wants 33.8%, savings 16.8% |
| Categoriser test accuracy | **96.5%** (macro F1 0.965, 5 mistakes out of 144) |
| Categoriser accuracy on demo transactions | 95.6% |
| Anomalies | 9 of 421 flagged; 7 of 7 planted caught |
| Forecast for November 2026 | ₹58,746 |
| Forecast backtest, mean error over 3 months | Linear Regression 12.3%; average baseline 5.6% |
| Recommendations generated | 8 |

Example tips:

- Wants are 34% of income; trimming ₹3,206 a month reaches 30%.
- Weekend wants spending is 1.8× the weekday level (₹985 vs ₹536 a day); a ₹669 daily weekend budget saves about ₹2,786 a month.
- Four streaming subscriptions cost ₹1,216 a month; keeping only Netflix saves ₹567.

Typical test mistakes are understandable. "swiggy intamart", a grocery app, was read as food delivery, and the typo "BILNKIT" was read as a bill.

[SCREENSHOT: Dashboard]
[SCREENSHOT: Add Transaction]
[SCREENSHOT: Upload CSV]
[SCREENSHOT: Spending Patterns]
[SCREENSHOT: Anomalies]
[SCREENSHOT: Forecast]
[SCREENSHOT: Recommendations]
[SCREENSHOT: AI Model]

## 11. Benefits

- Removes manual sorting, and shows its confidence so the user knows when to check.
- Turns raw transactions into clear charts and patterns.
- Catches unusual or mistaken charges early, with an understandable reason.
- Gives advice in rupees that is based on the user's own habits.
- Keeps data private: everything runs offline.

## 12. Practical Applications

- **Individuals and students:** budgeting and tracking subscriptions.
- **Banking and UPI apps:** automatic categorisation and "unusual spend" alerts.
- **Financial counsellors:** a quick 50/30/20 health check for clients.
- **Freelancers and small businesses:** planning for one-off costs and separating spending.
- **Expense-management tools:** pre-filling categories on expense reports.

## 13. Limitations & Future Scope

**Limitations**

- **Synthetic data:** real bank descriptions are messier, so real-world accuracy would likely be lower until the model is retrained on real data.
- **Unknown merchants:** the model is unsure about merchants it has never seen. For example, it guesses "Chai Point" as Transfers with only 12% confidence, so the user must correct it.
- **Forecast:** with six months of history, Linear Regression is less accurate on totals than a plain average, so it is mainly useful for showing direction.
- **Overlapping tips:** the tips are hand-written rules and their savings overlap, so they should not be added together.
- **No storage:** added data lasts only for the browser session.

**Future scope**

- Retrain on user corrections.
- Add longer history and seasonal models (for example Diwali shopping).
- Import bank PDF statements.
- Add budgets with alerts, savings goals and database storage.

## 14. How to Run

```bash
cd finwise-ai
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python data/generate_data.py    # optional: rebuilds identical sample data
python smoke_test.py            # trains and checks every model, prints accuracy
streamlit run app.py            # opens http://localhost:8501
```

## 15. Five-Minute Demo Script

| Time | Click | Say |
|---|---|---|
| 0:00 | **Dashboard** | "Six months of sample data: ₹5.02 lakh income, ₹4.18 lakh spent, a 16.8% savings rate, 3.2 points below the 20% target." |
| 0:40 | The two charts | "Monthly income vs expenses, and where the money goes; rent, shopping and food are the biggest." |
| 1:00 | **Add Transaction**, type "Zomato dinner order" | "As I type, the Logistic Regression model predicts Food & Dining live, with its confidence and the top 3." |
| 1:30 | Type "Jio recharge 666" | "Only 32% confidence, so the app warns me and I can change the category." Add it. |
| 2:00 | **Upload CSV**: download the example, then upload it | "These rows have no category; the model fills them in. Chai Point is new to it, so it guesses with just 12% confidence, which tells me to check that one." |
| 2:40 | **Spending Patterns** | "Month-over-month change, top merchants (Swiggy is the biggest habit) and weekends against weekdays." |
| 3:10 | **Anomalies** | "The Isolation Forest found all 7 unusual payments, for example an Uber at 12.8× my usual fare." |
| 3:40 | **Forecast** | "Linear Regression forecasts ₹58,746 for November and shows food rising about ₹581 a month. With six months of data, a plain average is still more accurate on totals." |
| 4:10 | **Recommendations** | "Eight tips from my own numbers, such as trimming wants by ₹3,206 a month." |
| 4:40 | **AI Model** | "96.5% accuracy on 144 unseen descriptions; this is the confusion matrix." |
| 5:00 | **Reset to sample data** | "Everything runs offline with fixed seeds, so the results never change." |
