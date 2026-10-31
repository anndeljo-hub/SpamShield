"""Stage 5: tune the three models with GridSearchCV (training data only).

Run from the project root:  python -m src.tune
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import load_clean_dataset

RANDOM_STATE = 42

# Parameters shared by all models (the vectorizer is part of the pipeline,
# so it is re-fitted inside each CV fold: no leakage).
TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
}

SEARCHES = {
    "MultinomialNB": (
        MultinomialNB(),
        {**TFIDF_GRID, "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE),
        {**TFIDF_GRID, "clf__C": [0.1, 1, 10, 100]},
    ),
    "LinearSVC": (
        LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        {**TFIDF_GRID, "clf__C": [0.1, 0.5, 1, 5, 10]},
    ),
}


def main():
    df = load_clean_dataset()
    # Same split as Stage 4, so results are directly comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"],
        test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, (clf, grid) in SEARCHES.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("clf", clf),
        ])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)

        pred = search.predict(X_test)  # test set used once, after tuning
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        print(f"\n{name}")
        print(f"  best params : {search.best_params_}")
        print(f"  CV spam F1  : {search.best_score_:.4f}  (training data only)")
        rows.append({
            "model": name,
            "cv_f1": search.best_score_,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "FP": fp, "FN": fn, "TP": tp,
        })

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())


if __name__ == "__main__":
    main()
