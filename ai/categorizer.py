"""
AI categorizer: predicts a transaction's category from its description.

How it works (two steps joined in one scikit-learn Pipeline):

1. TF-IDF turns each description into numbers. We use character n-grams
   (pieces of 2 to 4 letters inside each word), so "SWIGGY", "Swiggy" and the
   typo "Swigy" share most of their pieces and look similar to the model.
   Words that appear in every category (like "upi" or "order") get a low
   weight automatically - that is the "IDF" part.
2. Logistic Regression learns how strongly each piece points to each
   category. predict_proba() gives one probability per category; the highest
   one is the prediction and its probability is the model's "confidence".

The model is trained on data/training_data.csv. 20% of that file is held
back as a test set the model never sees, and the accuracy on it is reported.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
TRAINING_CSV = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"
INCOME_CATEGORIES = ["Salary", "Other Income"]  # every other category is an expense category


def clean_text(text):
    """Lower-case the description and keep only letters (numbers like order IDs carry no meaning)."""
    text = re.sub(r"[^a-z]+", " ", str(text).lower())
    return text.strip()


def extract_merchant(description):
    """Best-effort merchant name from a bank description.

    "UPI/Swiggy/412345678" -> "Swiggy", "Uber - Trip #4821" -> "Uber",
    "POS/Flipkart/Bengaluru" -> "Flipkart".
    """
    text = str(description).strip()
    text = re.sub(r"^(upi|pos|neft|imps|rtgs)\s*[/\-:]\s*", "", text, flags=re.I)
    text = re.sub(r"^(paid to|sent to|transfer to|payment to)\s+", "", text, flags=re.I)
    first_part = re.split(r"\s+-\s+|/|#|@|\|", text)[0]
    first_part = re.sub(r"\d+", "", first_part).strip(" -.:")
    return first_part or str(description).strip()


def build_pipeline():
    """Create the (untrained) TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                  ngram_range=(2, 4), sublinear_tf=True)),
        ("model", LogisticRegression(C=10, max_iter=2000, random_state=SEED)),
    ])


def train_categorizer(csv_path=TRAINING_CSV):
    """Train the categorizer and measure it on a held-out test set.

    Returns a dict with:
      model      - the trained pipeline (used by the app)
      accuracy   - accuracy on the 20% test set (0 to 1)
      labels     - category names, in the order used by the confusion matrix
      confusion  - confusion matrix (rows = true category, columns = predicted)
      report     - per-category precision / recall / F1 as a DataFrame
      mistakes   - test descriptions the model got wrong
      n_train, n_test - number of training and test examples
    """
    data = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data["description"], data["category"], test_size=0.2,
        stratify=data["category"], random_state=SEED)

    model = build_pipeline()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    labels = sorted(data["category"].unique())
    report = pd.DataFrame(classification_report(y_test, predicted, output_dict=True)).T
    mistakes = pd.DataFrame({"description": X_test, "actual": y_test, "predicted": predicted})
    mistakes = mistakes[mistakes["actual"] != mistakes["predicted"]]

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, predicted),
        "labels": labels,
        "confusion": confusion_matrix(y_test, predicted, labels=labels),
        "report": report.loc[labels, ["precision", "recall", "f1-score", "support"]],
        "mistakes": mistakes.reset_index(drop=True),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_categories(model, descriptions, types=None):
    """Predict a category and a confidence (0 to 1) for each description.

    If `types` ('income' / 'expense' per row) is given, an expense can only get an
    expense category and income only an income category. The confidence is still
    the model's own probability for the chosen category.
    """
    probabilities = model.predict_proba(list(descriptions))
    if types is not None:
        income_class = np.isin(model.classes_, INCOME_CATEGORIES)
        income_row = (np.asarray(types) == "income")[:, None]
        probabilities = np.where(income_class == income_row, probabilities, 0.0)
    best = probabilities.argmax(axis=1)
    return pd.DataFrame({
        "category": model.classes_[best],
        "confidence": probabilities[np.arange(len(best)), best],
    })


def top_predictions(model, description, k=3):
    """Return the k most likely categories for one description as (category, probability) pairs."""
    probabilities = model.predict_proba([description])[0]
    order = probabilities.argsort()[::-1][:k]
    return [(model.classes_[i], float(probabilities[i])) for i in order]


def categorize_missing(model, df):
    """Fill in missing categories with the model's prediction.

    Adds two columns:
      confidence     - the model's confidence for AI-filled rows (empty for the others)
      categorized_by - "AI" if the model chose the category, "Given" if it came with the data
    """
    df = df.copy()
    # A fully empty category column is read as numbers (all NaN); make it a text column first.
    df["category"] = df["category"].astype(object)
    missing = df["category"].isna()
    df["confidence"] = np.nan
    df["categorized_by"] = "Given"
    if missing.any():
        predictions = predict_categories(model, df.loc[missing, "description"], df.loc[missing, "type"])
        df.loc[missing, "category"] = predictions["category"].to_numpy()
        df.loc[missing, "confidence"] = predictions["confidence"].to_numpy()
        df.loc[missing, "categorized_by"] = "AI"
    return df


def accuracy_on(model, df):
    """Accuracy of the model on any labelled DataFrame (for example the demo transactions)."""
    labelled = df.dropna(subset=["category"])
    return accuracy_score(labelled["category"], model.predict(labelled["description"]))
