"""Load the saved model and predict one SMS, with word-level explanation.

Run a quick demo:  python -m src.predict
"""

import joblib

from src.data_loader import PROJECT_ROOT
from src.preprocessing import clean_text

MODEL_PATH = PROJECT_ROOT / "models" / "spam_detector.pkl"

_model = None


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run 'python -m src.train' first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_message(text, top_n=5):
    """Return a dict: label, spam_probability, confidence, top_spam_words,
    top_ham_words. Raises ValueError for empty input."""
    cleaned = clean_text(text)
    if cleaned == "":
        raise ValueError("Message is empty.")

    model = load_model()
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    spam_prob = float(model.predict_proba([cleaned])[0, 1])
    label = "spam" if spam_prob >= 0.5 else "ham"
    confidence = spam_prob if label == "spam" else 1 - spam_prob

    # Word contributions = tfidf value * coefficient (only words in this message)
    vec = tfidf.transform([cleaned])
    names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    contribs = []
    row = vec.tocoo()
    for idx, val in zip(row.col, row.data):
        contribs.append((names[idx], float(val * coefs[idx])))

    spam_words = sorted([c for c in contribs if c[1] > 0],
                        key=lambda c: -c[1])[:top_n]
    ham_words = sorted([c for c in contribs if c[1] < 0],
                       key=lambda c: c[1])[:top_n]

    return {
        "label": label,
        "spam_probability": spam_prob,
        "confidence": confidence,
        "top_spam_words": spam_words,
        "top_ham_words": ham_words,
    }


if __name__ == "__main__":
    demos = [
        "Congratulations! You won a FREE prize. Call 09061701461 now to claim your cash!",
        "Hey, are we still meeting for lunch at 1?",
        "Your OTP is 482913. Do not share it with anyone.",
    ]
    for msg in demos:
        r = predict_message(msg)
        print(f"\n{msg}")
        print(f"  -> {r['label'].upper()}  spam_prob={r['spam_probability']:.3f}")
        print(f"  spam signals: {r['top_spam_words']}")
        print(f"  ham signals : {r['top_ham_words']}")
