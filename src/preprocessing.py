"""Reusable text preprocessing for SpamShield.

The SAME function is used in training and in the app, so the model always
sees text cleaned the same way (avoids train/serve skew).
"""

import html
import re

import pandas as pd

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
URL_PLACEHOLDER = re.compile(r"<url>", re.IGNORECASE)
NUM_PLACEHOLDER = re.compile(r"<#>")
MONEY_SYMBOLS = re.compile(r"[£$€]")
WHITESPACE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalise one SMS message.

    Steps (and why):
      1. Handle None/NaN/non-strings -> empty string (app input can be odd).
      2. Decode HTML entities (&lt; -> <) so placeholders become readable.
      3. Lowercase so 'FREE' and 'free' are one feature.
      4. Map placeholders and real URLs to single tokens (urltoken) so the
         model learns "a link is present" instead of memorising each URL.
      5. Map '<#>' to numtoken.
      6. Turn currency symbols into a word (moneysymbol); spam loves prizes.
      7. Collapse repeated whitespace.

    Deliberately KEPT: digits, '!' and other punctuation, because the EDA
    showed digits appear in 94.8% of spam vs 15.7% of ham.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = text.lower()
    text = URL_PLACEHOLDER.sub(" urltoken ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = NUM_PLACEHOLDER.sub(" numtoken ", text)
    text = MONEY_SYMBOLS.sub(" moneysymbol ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def is_empty_message(text) -> bool:
    """True if the message has no usable content after cleaning."""
    return clean_text(text) == ""


def load_clean_dataset(drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the dataset, drop exact duplicates, add 'clean' text and a
    numeric 'target' (spam=1, ham=0)."""
    from src.data_loader import load_data

    df = load_data()
    before = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows before dedup: {before}, after: {len(df)}")

    df["clean"] = df["message"].apply(clean_text)
    df = df[df["clean"] != ""].reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


if __name__ == "__main__":
    samples = [
        "WINNER!! You won £1000 cash. Call 09061701461 or visit www.prize.com NOW",
        "I am  &lt;#&gt;  inches...",
        "Find out now @  &lt;URL&gt;",
        "   ",
        None,
    ]
    for s in samples:
        print(repr(s), "->", repr(clean_text(s)))

    data = load_clean_dataset()
    print(data["label"].value_counts())
    print(data[["label", "clean"]].head())
