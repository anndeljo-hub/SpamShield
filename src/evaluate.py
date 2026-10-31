"""Stage 6: repeated-CV comparison + ROC/PR analysis of the tuned models.

Run from the project root:  python -m src.evaluate
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import PROJECT_ROOT
from src.preprocessing import load_clean_dataset

RANDOM_STATE = 42


def pipe(clf, min_df):
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                  min_df=min_df)),
        ("clf", clf),
    ])


# Settings found by GridSearchCV in Stage 5.
MODELS = {
    "MultinomialNB": pipe(MultinomialNB(alpha=0.01), min_df=1),
    "LogisticRegression": pipe(LogisticRegression(
        C=10, max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE), min_df=2),
    "LinearSVC": pipe(LinearSVC(
        C=0.5, class_weight="balanced", random_state=RANDOM_STATE), min_df=2),
}


def spam_score(model, X):
    """Continuous spam score, whatever the model offers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    df = load_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"],
        test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE,
    )

    # --- Repeated CV on TRAINING data only ---
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                 random_state=RANDOM_STATE)
    rows = []
    for name, model in MODELS.items():
        res = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                             scoring=["precision", "recall", "f1"])
        rows.append({
            "model": name,
            "cv_precision": res["test_precision"].mean(),
            "cv_recall": res["test_recall"].mean(),
            "cv_f1": res["test_f1"].mean(),
            "cv_f1_std": res["test_f1"].std(),
        })
    print("=== REPEATED CV (15 folds, training data only) ===")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())

    # --- Threshold-free metrics on the TEST set ---
    plt.figure(figsize=(6, 5))
    rows = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        score = spam_score(model, X_test)
        rows.append({
            "model": name,
            "test_roc_auc": roc_auc_score(y_test, score),
            "test_pr_auc": average_precision_score(y_test, score),
        })
        p, r, _ = precision_recall_curve(y_test, score)
        plt.plot(r, p, label=name)
    print("\n=== TEST SET: threshold-free metrics ===")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())

    plt.xlabel("Recall (spam)")
    plt.ylabel("Precision (spam)")
    plt.title("Precision-recall curves (test set)")
    plt.legend()
    plt.ylim(0.5, 1.02)
    plt.tight_layout()
    out = PROJECT_ROOT / "reports" / "figures" / "pr_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
