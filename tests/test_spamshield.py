"""Tests for preprocessing and prediction. Run:  python -m pytest -v"""

import pytest

from src.preprocessing import clean_text, is_empty_message
from src.predict import predict_message


# ---------- Preprocessing ----------

def test_empty_and_none_become_empty_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert is_empty_message("   ")


def test_lowercases_text():
    assert clean_text("HELLO World") == "hello world"


def test_html_entities_decoded_to_tokens():
    assert clean_text("I am &lt;#&gt; inches") == "i am numtoken inches"
    assert clean_text("see &lt;URL&gt;") == "see urltoken"


def test_urls_become_token():
    assert "urltoken" in clean_text("visit www.prize.com now")
    assert "urltoken" in clean_text("go to http://example.com/win")
    assert "prize.com" not in clean_text("visit www.prize.com now")


def test_numbers_and_punctuation_are_kept():
    out = clean_text("Call 09061701461 now!!")
    assert "09061701461" in out
    assert "!!" in out


def test_currency_symbol_mapped():
    assert "moneysymbol" in clean_text("You won £1000")


def test_non_string_input_does_not_crash():
    assert clean_text(12345) == "12345"


# ---------- Prediction ----------

def test_empty_message_raises():
    with pytest.raises(ValueError):
        predict_message("   ")


def test_typical_spam_is_spam():
    r = predict_message(
        "Congratulations! You won a FREE prize. Call 09061701461 now to claim your cash!")
    assert r["label"] == "spam"
    assert r["spam_probability"] > 0.9


def test_normal_message_is_ham():
    r = predict_message("Hey, are we still meeting for lunch at 1?")
    assert r["label"] == "ham"
    assert r["spam_probability"] < 0.5


def test_message_with_url_is_valid_output():
    # A URL alone must not be assumed to mean spam: we only check validity.
    r = predict_message("Here is the notes link www.example.com see you tomorrow")
    assert r["label"] in {"spam", "ham"}
    assert 0.0 <= r["spam_probability"] <= 1.0


def test_message_with_numbers_is_valid_output():
    r = predict_message("Your OTP is 482913. Do not share it with anyone.")
    assert r["label"] in {"spam", "ham"}
    assert 0.0 <= r["spam_probability"] <= 1.0


def test_unseen_message_returns_complete_result():
    r = predict_message("Meeting moved to the library, bring your laptop")
    assert set(r) == {"label", "spam_probability", "confidence",
                      "top_spam_words", "top_ham_words"}
    assert 0.5 <= r["confidence"] <= 1.0


def test_probability_and_label_consistent():
    r = predict_message("WINNER!! Claim your free cash prize now")
    assert (r["label"] == "spam") == (r["spam_probability"] >= 0.5)
