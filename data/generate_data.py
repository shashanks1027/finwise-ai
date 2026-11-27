"""
FinWise AI - sample data generator
==================================
Creates two CSV files in this folder:

* transactions.csv  - six months (May to October 2026) of one person's bank
                      transactions. This is the demo data the app loads.
* training_data.csv - labelled transaction descriptions used to train the
                      AI categorizer (see ai/categorizer.py).

Persona: a young software engineer in Bengaluru earning Rs 78,000 a month.

A fixed random seed (SEED) is used, so every run writes exactly the same files.

Run from the finwise-ai folder:
    python data/generate_data.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).resolve().parent
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
TRAINING_CSV = DATA_DIR / "training_data.csv"

YEAR = 2026
MONTHS = [5, 6, 7, 8, 9, 10]  # May to October 2026
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SALARY = 78000

CITIES = ["Bengaluru", "Mumbai", "Pune", "Hyderabad", "Chennai", "Delhi", "Gurugram", "Noida"]
UPI_BANKS = ["ybl", "okicici", "okhdfcbank", "paytm", "axl", "ibl", "okaxis", "oksbi"]
PEOPLE = ["Rahul Sharma", "Priya Nair", "Amit Verma", "Sneha Iyer", "Rohan Gupta",
          "Ananya Reddy", "Vikram Singh", "Kavya Menon", "Arjun Rao", "Neha Joshi",
          "Karthik Subramanian", "Pooja Patel", "Aditya Kulkarni", "Meera Pillai"]

# ---------------------------------------------------------------------------
# Merchants and details for every category (used to build training data).
# Some names deliberately appear in more than one category (Amazon, Swiggy,
# Jio, Tata, Flipkart) so the model has to learn from the other words too.
# ---------------------------------------------------------------------------
MERCHANTS = {
    "Food & Dining": ["Swiggy", "Zomato", "Domino's Pizza", "McDonald's", "KFC", "Starbucks",
                      "Cafe Coffee Day", "Chaayos", "Haldiram's", "Burger King", "Pizza Hut",
                      "Barbeque Nation", "Third Wave Coffee", "Behrouz Biryani", "Meghana Foods",
                      "Subway", "EatFit", "Truffles", "Paradise Biryani", "A2B Restaurant"],
    "Groceries": ["BigBasket", "Blinkit", "Zepto", "Swiggy Instamart", "DMart", "Reliance Fresh",
                  "More Supermarket", "Amazon Fresh", "JioMart", "Nature's Basket",
                  "Country Delight", "Namdhari's Fresh", "Spencer's", "Kirana Store",
                  "Milkbasket", "Ratnadeep Supermarket"],
    "Transport": ["Uber", "Ola", "Rapido", "Namma Metro", "IRCTC", "HP Petrol Pump", "Indian Oil",
                  "Bharat Petroleum", "FASTag", "RedBus", "Uber Moto", "Ola Auto", "Yulu",
                  "BMTC", "MakeMyTrip", "IndiGo", "Delhi Metro", "Mumbai Local"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Meesho", "Nykaa", "Croma",
                 "Reliance Digital", "Decathlon", "Lifestyle", "Westside", "IKEA", "Tata CLiQ",
                 "Lenskart", "Zara", "H&M", "Pantaloons", "boAt"],
    "Bills & Utilities": ["BESCOM", "Tata Power", "Jio Prepaid", "Airtel Postpaid", "ACT Fibernet",
                          "Indane Gas", "BWSSB", "Vi Prepaid", "Airtel Xstream Fiber",
                          "Society Maintenance", "Bharat Gas", "Tata Play", "MSEDCL", "Adani Electricity"],
    "Entertainment": ["Netflix", "Spotify", "Disney+ Hotstar", "Amazon Prime Video", "YouTube Premium",
                      "BookMyShow", "PVR Cinemas", "INOX", "Steam", "SonyLIV", "JioCinema Premium",
                      "Wonderla", "Gaana Plus", "Zee5", "PlayStation Store"],
    "Rent": ["House Rent", "Monthly Rent", "PG Rent", "Flat Rent", "NoBroker Rent", "Room Rent",
             "Rent Payment", "Landlord", "Hostel Rent", "Rent"],
    "Health": ["Apollo Pharmacy", "PharmEasy", "Tata 1mg", "MedPlus", "Practo", "Cult.fit",
               "Apollo Hospitals", "Thyrocare", "Netmeds", "Manipal Hospitals", "Clove Dental",
               "Dr Lal PathLabs", "Gold's Gym", "HealthifyMe"],
    "Education": ["Udemy", "Coursera", "Unacademy", "BYJU'S", "College Fee", "Exam Fee",
                  "Sapna Book House", "Scaler Academy", "upGrad", "Simplilearn", "Stationery Shop",
                  "Kindle Store", "Physics Wallah", "NPTEL"],
    "Transfers": PEOPLE,
    "Salary": ["Acme Technologies Pvt Ltd", "Infosys Ltd", "TCS", "Wipro Ltd", "Accenture",
               "Zoho Corp", "Freshworks", "Flipkart Internet Pvt Ltd", "HCL Technologies",
               "Tech Mahindra", "Cognizant", "Employer"],
    "Other Income": ["Upwork", "Fiverr", "HDFC Bank", "SBI", "ICICI Bank", "Amazon Pay", "Paytm",
                     "Google Pay", "Zerodha", "Income Tax Dept", "Freelance Client", "PhonePe"],
}

DETAILS = {
    "Food & Dining": ["Order #{n}", "Dinner", "Lunch", "Food order", "Breakfast", "Snacks",
                      "Coffee", "Takeaway", "Koramangala", "Indiranagar"],
    "Groceries": ["Order #{n}", "Groceries", "Vegetables and fruits", "Monthly groceries",
                  "Milk and bread", "Household items", "Daily essentials"],
    "Transport": ["Trip #{n}", "Ride", "Airport ride", "Office commute", "Card recharge", "Fuel",
                  "Ticket", "Booking #{n}"],
    "Shopping": ["Order #{n}", "Clothes", "Electronics", "Shoes", "Home decor", "Gift", "Accessories"],
    "Bills & Utilities": ["Electricity bill", "Recharge {n}", "Monthly bill", "Bill payment",
                          "Cylinder refill", "Broadband bill", "Water bill", "Postpaid bill"],
    "Entertainment": ["Monthly subscription", "Premium plan", "Movie tickets", "Subscription renewal",
                      "Game purchase", "Annual plan", "Show tickets"],
    "Rent": ["{Mon} 2026", "Flat 302", "To landlord", "Monthly", "Advance", "Rent for {Mon}"],
    "Health": ["Medicines", "Doctor consultation", "Membership", "Lab test", "Health checkup",
               "Pharmacy order #{n}", "Treatment"],
    "Education": ["Course", "Online course", "Books", "Semester fee", "Registration", "Certification",
                  "Subscription"],
    "Transfers": ["Trip split", "Dinner split", "Cab share", "Birthday gift", "Movie split",
                  "Returned money", "Loan repay"],
    "Salary": ["Salary credit", "Salary {Mon} 2026", "Monthly salary", "Payroll credit", "SAL CREDIT"],
    "Other Income": ["Freelance payment", "Interest credit", "Cashback received", "Refund",
                     "Dividend credit", "Rewards", "Tax refund", "Payout"],
}

# Description layouts seen on Indian bank statements. Placeholders:
# {m} merchant, {M} merchant in capitals, {d} detail, {D} detail in capitals,
# {ref} reference number, {city}/{CITY}, {handle}@{bank} a UPI id.
SHOP_TEMPLATES = ["{m}", "{m} - {d}", "UPI/{m}/{ref}", "UPI-{M}-{handle}@{bank}", "POS/{M}/{CITY}",
                  "POS {m} {city}", "Paid to {m}", "{d} at {m}", "{m} {d}", "{M} {D}"]
INCOME_TEMPLATES = ["{d} - {m}", "{m} - {d}", "NEFT/{m}/{d}", "IMPS/{m}/{ref}", "{M} {D}",
                    "{d} from {m}"]
TRANSFER_TEMPLATES = ["UPI/{m}/{d}", "Sent to {m}", "Transfer to {m} - {d}", "IMPS/{m}/{ref}",
                      "UPI-{M}-{handle}@{bank}", "Paid to {m}", "{d} - {m}"]
RENT_TEMPLATES = ["{m}", "{m} - {d}", "UPI/{m}/{d}", "NEFT/{m}/{ref}", "{M} {D}", "Paid to {m}",
                  "{m} {d}"]
TEMPLATES = {"Salary": INCOME_TEMPLATES, "Other Income": INCOME_TEMPLATES,
             "Transfers": TRANSFER_TEMPLATES, "Rent": RENT_TEMPLATES}

# ---------------------------------------------------------------------------
# Spending patterns for the six months of demo transactions.
# Each group: (merchants, details, lowest amount, highest amount, share of transactions)
# ---------------------------------------------------------------------------
TXN_GROUPS = {
    "Food & Dining": [
        (["Swiggy", "Zomato"], ["Order #{n}"], 180, 750, 0.6),
        (["Starbucks", "Cafe Coffee Day", "Chaayos", "Third Wave Coffee"], ["Coffee", "Snacks"], 120, 480, 0.2),
        (["Meghana Foods", "Truffles", "Domino's Pizza", "McDonald's", "Behrouz Biryani", "Barbeque Nation"],
         ["Dinner", "Lunch", "Koramangala", "Indiranagar"], 250, 1800, 0.2),
    ],
    "Groceries": [
        (["Blinkit", "Zepto", "Swiggy Instamart"], ["Order #{n}"], 150, 900, 0.65),
        (["BigBasket", "DMart", "Reliance Fresh"], ["Monthly groceries", "Groceries", "Vegetables and fruits"],
         900, 2200, 0.35),
    ],
    "Transport": [
        (["Uber", "Ola", "Rapido"], ["Trip #{n}", "Ride"], 70, 480, 0.8),
        (["Namma Metro"], ["Card recharge"], 300, 600, 0.1),
        (["Indian Oil", "HP Petrol Pump"], ["Fuel"], 400, 1200, 0.1),
    ],
    "Shopping": [
        (["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "Decathlon", "Lenskart"], ["Order #{n}"], 199, 4000, 1.0),
    ],
    "Entertainment": [
        (["BookMyShow", "PVR Cinemas", "INOX"], ["Movie tickets"], 300, 1100, 1.0),
    ],
    "Health": [
        (["Apollo Pharmacy", "PharmEasy", "Tata 1mg", "MedPlus"], ["Medicines"], 120, 900, 0.8),
        (["Practo"], ["Doctor consultation"], 400, 800, 0.2),
    ],
    "Education": [
        (["Udemy"], ["Python course", "Data science course"], 399, 599, 0.6),
        (["Sapna Book House"], ["Books"], 250, 900, 0.4),
    ],
    "Transfers": [
        (PEOPLE[:6], ["Trip split", "Dinner split", "Cab share", "Birthday gift", "Movie split"], 200, 2500, 1.0),
    ],
}

# How many transactions of each kind per month (lowest, highest).
COUNT_RANGE = {"Food & Dining": (16, 20), "Groceries": (8, 11), "Transport": (14, 19), "Shopping": (3, 6),
               "Entertainment": (1, 3), "Health": (1, 3), "Education": (0, 2), "Transfers": (2, 4)}

# Weekend days are this many times more likely than weekdays for these categories.
WEEKEND_WEIGHT = {"Food & Dining": 2.0, "Shopping": 1.6, "Entertainment": 3.0, "Transfers": 1.5}

# Electricity bill per month (higher in the hot months of May and June).
ELECTRICITY = [1850, 1700, 1350, 1250, 1200, 1300]

# Deliberately unusual expenses so the anomaly detector has something real to find.
PLANTED_ANOMALIES = [
    ("2026-05-27", "BESCOM - Electricity bill arrears", 5240, "Bills & Utilities"),
    ("2026-06-14", "Croma - Sony WH-1000XM5 headphones", 18990, "Shopping"),
    ("2026-07-22", "Apollo Hospitals - Emergency treatment", 12450, "Health"),
    ("2026-08-15", "Zomato - Order #88231", 4860, "Food & Dining"),
    ("2026-09-05", "Uber - Airport ride", 2350, "Transport"),
    ("2026-10-03", "MakeMyTrip - Flight BLR-DEL", 8740, "Transport"),
    ("2026-10-19", "Amazon - Order #40917", 9999, "Shopping"),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def pick(rng, items):
    """Return one random element of a list (as a normal Python object)."""
    return items[rng.integers(len(items))]


def random_amount(rng, low, high):
    """Random amount between low and high. Small amounts are more common than big ones."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def random_day(rng, month, weekend_weight=1.0):
    """Pick a random date in the month; weekends get extra weight if weekend_weight > 1."""
    days = pd.date_range(f"{YEAR}-{month:02d}-01", periods=pd.Period(f"{YEAR}-{month:02d}").days_in_month)
    weights = np.where(days.day_of_week >= 5, weekend_weight, 1.0)
    return days[rng.choice(len(days), p=weights / weights.sum())]


def fill_detail(rng, detail):
    """Replace {n} (an order number) and {Mon} (a month name) inside a detail phrase."""
    detail = detail.replace("{n}", str(rng.integers(1000, 99999)))
    return detail.replace("{Mon}", pick(rng, MONTH_NAMES))


def add_typo(rng, word):
    """Make a realistic typo: drop one letter or swap two neighbouring letters."""
    if len(word) < 5:
        return word
    i = int(rng.integers(1, len(word) - 2))
    if rng.random() < 0.5:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def fill_template(rng, template, merchant, detail, city="Bengaluru"):
    """Turn a template such as 'UPI/{m}/{ref}' into a real description."""
    return template.format(
        m=merchant, M=merchant.upper(), d=detail, D=detail.upper(),
        ref=str(rng.integers(100_000_000, 999_999_999)),
        city=city, CITY=city.upper(),
        handle=re.sub(r"[^a-z0-9]", "", merchant.lower()), bank=pick(rng, UPI_BANKS),
    )


# ---------------------------------------------------------------------------
# transactions.csv
# ---------------------------------------------------------------------------
def fixed_monthly_rows(rng, i, month):
    """Salary, rent, bills and subscriptions: the things that happen every month."""
    day = lambda d: pd.Timestamp(YEAR, month, d)  # noqa: E731 - tiny local helper
    last_day = pd.Period(f"{YEAR}-{month:02d}").days_in_month
    rows = [
        (day(1), "Salary Credit - Acme Technologies Pvt Ltd", SALARY, "income", "Salary"),
        (day(5), "UPI/House Rent/Landlord Ramesh Kumar", 18000, "expense", "Rent"),
        (day(8), f"BESCOM - Electricity bill {MONTH_NAMES[month - 1]} {YEAR}",
         ELECTRICITY[i] + int(rng.integers(-80, 80)), "expense", "Bills & Utilities"),
        (day(10), "Society Maintenance - Prestige Apartments", 1200, "expense", "Bills & Utilities"),
        (day(12), "Jio Prepaid - Recharge 349", 349, "expense", "Bills & Utilities"),
        (day(15), "ACT Fibernet - Broadband bill", 799, "expense", "Bills & Utilities"),
        # Recurring subscriptions (same text and amount every month)
        (day(2), "Cult.fit - Monthly membership", 1299, "expense", "Health"),
        (day(7), "Netflix - Monthly subscription", 649, "expense", "Entertainment"),
        (day(18), "Spotify - Premium Individual", 119, "expense", "Entertainment"),
        (day(25), "YouTube Premium - Monthly", 149, "expense", "Entertainment"),
    ]
    if i >= 1:  # Hotstar was taken from June onwards
        rows.append((day(21), "Disney+ Hotstar - Super plan", 299, "expense", "Entertainment"))
    if i % 2 == 0:  # gas cylinder every second month
        rows.append((day(20), "Indane Gas - Cylinder refill", 905, "expense", "Bills & Utilities"))
    if i in (1, 4):  # savings interest is credited quarterly (June, September)
        rows.append((day(last_day), "Interest Credit - HDFC Bank", int(rng.integers(450, 650)),
                     "income", "Other Income"))
    if rng.random() < 0.6:  # freelance work in some months
        rows.append((random_day(rng, month), "Upwork - Freelance payment",
                     int(rng.integers(10, 25)) * 500, "income", "Other Income"))
    if rng.random() < 0.5:
        rows.append((random_day(rng, month), "Cashback - Amazon Pay", int(rng.integers(25, 150)),
                     "income", "Other Income"))
    return rows


def variable_rows(rng, i, month):
    """Day-to-day spending: food, groceries, rides, shopping and so on."""
    rows = []
    for category, groups in TXN_GROUPS.items():
        low, high = COUNT_RANGE[category]
        count = int(rng.integers(low, high + 1))
        amount_scale = 1.0
        if category == "Food & Dining":  # food spending creeps up every month
            count = round(count * (1 + 0.06 * i))
            amount_scale = 1 + 0.04 * i
        if category == "Shopping" and month in (9, 10):  # festive sales
            count += 2
        shares = np.array([g[4] for g in groups])
        for _ in range(count):
            merchants, details, lo, hi, _share = groups[rng.choice(len(groups), p=shares / shares.sum())]
            merchant = pick(rng, merchants)
            detail = fill_detail(rng, pick(rng, details))
            if category == "Transfers":
                text = f"UPI/{merchant}/{detail}"
            else:
                text = fill_template(rng, pick(rng, ["{m} - {d}", "{m} - {d}", "UPI/{m}/{ref}", "POS/{m}/{city}"]),
                                     merchant, detail)
            amount = round(random_amount(rng, lo, hi) * amount_scale)
            date = random_day(rng, month, WEEKEND_WEIGHT.get(category, 1.0))
            rows.append((date, text, amount, "expense", category))
    return rows


def build_transactions(seed=SEED):
    """Build the six months of demo transactions as a DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, month in enumerate(MONTHS):
        rows += fixed_monthly_rows(rng, i, month)
        rows += variable_rows(rng, i, month)
    for date, text, amount, category in PLANTED_ANOMALIES:
        rows.append((pd.Timestamp(date), text, amount, "expense", category))

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type", "category"])
    df = df.sort_values(["date", "type", "description"], kind="mergesort").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["amount"] = df["amount"].astype(int)
    return df


# ---------------------------------------------------------------------------
# training_data.csv
# ---------------------------------------------------------------------------
def training_description(rng, category):
    """One random, realistic-looking description for the given category."""
    merchant = pick(rng, MERCHANTS[category])
    if rng.random() < 0.15:
        merchant = add_typo(rng, merchant)
    detail = fill_detail(rng, pick(rng, DETAILS[category]))
    template = pick(rng, TEMPLATES.get(category, SHOP_TEMPLATES))
    text = fill_template(rng, template, merchant, detail, city=pick(rng, CITIES))
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    elif roll < 0.30:
        text = text.upper()
    return text


def build_training_data(seed=SEED + 1, per_category=60):
    """Build `per_category` distinct labelled descriptions for every category.

    Two descriptions that differ only in numbers or punctuation (for example
    two Swiggy order numbers) count as the same, so the test set is not full
    of near-copies of the training set.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for category in MERCHANTS:
        seen_keys = set()  # used only for membership checks; order comes from `rows`
        tries = 0
        while len(seen_keys) < per_category and tries < per_category * 100:
            tries += 1
            text = training_description(rng, category)
            key = re.sub(r"[^a-z]+", " ", text.lower()).strip()
            if key not in seen_keys:
                seen_keys.add(key)
                rows.append((text, category))
    df = pd.DataFrame(rows, columns=["description", "category"])
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def main():
    """Write both CSV files and print a short summary."""
    transactions = build_transactions()
    training = build_training_data()
    transactions.to_csv(TRANSACTIONS_CSV, index=False, lineterminator="\n")
    training.to_csv(TRAINING_CSV, index=False, lineterminator="\n")

    income = transactions.loc[transactions["type"] == "income", "amount"].sum()
    expense = transactions.loc[transactions["type"] == "expense", "amount"].sum()
    print(f"transactions.csv : {len(transactions)} rows "
          f"({transactions['date'].min()} to {transactions['date'].max()})")
    print(f"                   income Rs {income:,} | expenses Rs {expense:,}")
    print(f"training_data.csv: {len(training)} rows, {training['category'].nunique()} categories")


if __name__ == "__main__":
    main()
