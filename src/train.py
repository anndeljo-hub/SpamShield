"""Train the final SpamShield model on the training split and save it.

Run from the project root:  python -m src.train
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT
from src.preprocessing import load_clean_dataset

RANDOM_STATE = 42
MODEL_PATH = PROJECT_ROOT / "models" / "spam_detector.pkl"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                  min_df=2)),
        ("clf", LogisticRegression(C=10, max_iter=2000,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def main():
    df = load_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"],
        test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE,
    )
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("=== TRAIN performance ===")
    print(classification_report(y_train, model.predict(X_train),
                                target_names=["ham", "spam"], digits=4))
    print("=== TEST performance (honest estimate) ===")
    pred = model.predict(X_test)
    print(classification_report(y_test, pred,
                                target_names=["ham", "spam"], digits=4))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
