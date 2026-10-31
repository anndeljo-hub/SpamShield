"""Exploratory data analysis for the SMS Spam Collection.

Run from the project root:  python -m src.eda
Saves charts to reports/figures/ and prints the numbers behind them.
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import PROJECT_ROOT, load_data

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Small stop-word list, only for the word-frequency chart.
# (Stage 3 decides what the real pipeline does.)
STOP_WORDS = {
    "i", "you", "to", "the", "a", "and", "in", "is", "me", "my", "it", "u",
    "your", "for", "of", "on", "that", "have", "are", "at", "so", "but",
    "be", "we", "not", "do", "can", "will", "if", "or", "with", "just",
    "this", "ur", "2", "its", "im", "am", "get", "go", "now", "was", "up",
    "ok", "all", "out", "when", "what", "how", "from", "no", "then", "as",
}


def top_words(messages, n=15):
    """Return the n most frequent non-stop-words in a list of messages."""
    counter = Counter()
    for msg in messages:
        words = re.findall(r"[a-z0-9']+", msg.lower())
        counter.update(w for w in words if w not in STOP_WORDS and len(w) > 1)
    return counter.most_common(n)


def save(fig_name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close()
    print(f"  saved reports/figures/{fig_name}")


def main():
    df = load_data()
    sns.set_theme(style="whitegrid")

    # --- 1. Duplicates by class ---
    dup_mask = df.duplicated()
    print("=== DUPLICATES BY CLASS ===")
    print(df[dup_mask]["label"].value_counts())
    conflicts = df.groupby("message")["label"].nunique()
    print(f"\nMessages that appear with BOTH labels: {(conflicts > 1).sum()}")

    # --- 2. Class distribution ---
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="label", order=["ham", "spam"],
                       hue="label", palette=["#2E86AB", "#E4572E"], legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Class distribution")
    save("class_distribution.png")

    # --- 3. Message length ---
    df["char_len"] = df["message"].str.len()
    df["word_len"] = df["message"].str.split().str.len()
    print("\n=== MESSAGE LENGTH (characters) BY CLASS ===")
    print(df.groupby("label")["char_len"].describe().round(1))

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="char_len", hue="label", bins=40,
                 palette={"ham": "#2E86AB", "spam": "#E4572E"},
                 element="step", stat="density", common_norm=False)
    plt.title("Message length (characters), normalised per class")
    save("message_length.png")

    # --- 4. Top words per class ---
    for label, color in [("spam", "#E4572E"), ("ham", "#2E86AB")]:
        words = top_words(df[df["label"] == label]["message"])
        print(f"\n=== TOP WORDS: {label.upper()} ===")
        print(words)
        names, counts = zip(*words)
        plt.figure(figsize=(7, 5))
        sns.barplot(x=list(counts), y=list(names), color=color)
        plt.title(f"Most frequent words in {label}")
        plt.xlabel("Count")
        save(f"top_words_{label}.png")

    # --- 5. Simple spam signals ---
    df["has_digit"] = df["message"].str.contains(r"\d")
    df["has_url"] = df["message"].str.contains(r"http|www\.", case=False, regex=True)
    print("\n=== SHARE OF MESSAGES CONTAINING... (by class) ===")
    print(df.groupby("label")[["has_digit", "has_url"]].mean().round(3))


if __name__ == "__main__":
    main()
