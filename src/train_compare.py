"""Stage 4: compare MultinomialNB, Logistic Regression and Linear SVM.

Run from the project root:  python -m src.train_compare
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import load_clean_dataset

RANDOM_STATE = 42


def make_pipeline(classifier):
    # TF-IDF with unigrams + bigrams. The vectorizer lives INSIDE the
    # pipeline, so it is fitted on training data only (no leakage).
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)),
        ("clf", classifier),
    ])


def main():
    df = load_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"],
        test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)} (spam {y_train.mean():.3%})")
    print(f"Test:  {len(X_test)} (spam {y_test.mean():.3%})\n")

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    rows = []
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "spam_precision": precision_score(y_test, pred),
            "spam_recall": recall_score(y_test, pred),
            "spam_f1": f1_score(y_test, pred),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    results = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200)
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()
