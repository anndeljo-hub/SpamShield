"""SpamShield - Streamlit web app.  Run:  python -m streamlit run app.py"""

import streamlit as st

from src.predict import ModelNotFoundError, load_model, predict_message

st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .result-box {padding: 1.2rem 1.5rem; border-radius: 12px;
                 color: white; font-size: 1.6rem; font-weight: 700;}
    .spam {background: #C0392B;}
    .ham  {background: #1E8449;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Prize scam": "Congratulations! You won a FREE prize. Call 09061701461 now to claim your cash!",
    "Casual chat": "Hey, are we still meeting for lunch at 1?",
    "OTP message": "Your OTP is 482913. Do not share it with anyone.",
    "Reminder": "Reminder: your appointment is tomorrow at 10am. Reply C to confirm.",
}

if "message" not in st.session_state:
    st.session_state.message = ""

st.title("🛡️ SpamShield")
st.caption("Intelligent SMS spam detection using TF-IDF and Logistic Regression")

# Fail early and clearly if the model file is missing.
try:
    load_model()
except ModelNotFoundError as e:
    st.error(f"{e}")
    st.stop()

tab_detect, tab_model, tab_method = st.tabs(["Detector", "Model info", "Methodology"])

with tab_detect:
    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.message = text

    message = st.text_area("SMS message", key="message", height=150,
                           placeholder="Paste or type an SMS message here...")

    if st.button("Predict Spam", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            r = predict_message(message)
            is_spam = r["label"] == "spam"
            css = "spam" if is_spam else "ham"
            text = "🚨 SPAM" if is_spam else "✅ HAM (legitimate)"
            st.markdown(f'<div class="result-box {css}">{text}</div>',
                        unsafe_allow_html=True)

            st.write("")
            st.write(f"**Spam probability: {r['spam_probability']:.1%}**"
                     f"  |  Confidence in this label: {r['confidence']:.1%}")
            st.progress(r["spam_probability"])

            if 0.3 <= r["spam_probability"] <= 0.7:
                st.info("The model is not very sure about this one. "
                        "Treat the result with extra caution.")

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Words pushing toward spam**")
                if r["top_spam_words"]:
                    for w, s in r["top_spam_words"]:
                        st.write(f"- `{w}` ({s:+.2f})")
                else:
                    st.write("None found.")
            with c2:
                st.write("**Words pushing toward ham**")
                if r["top_ham_words"]:
                    for w, s in r["top_ham_words"]:
                        st.write(f"- `{w}` ({s:+.2f})")
                else:
                    st.write("None found.")

            st.caption("These are model-based signals showing what this model "
                       "learned from its training data. They are not guaranteed "
                       "causal explanations or proof that a message is spam.")

with tab_model:
    st.subheader("Model information")
    st.markdown(
        """
- **Algorithm:** Logistic Regression (C=10, balanced class weights)
- **Features:** TF-IDF, unigrams and bigrams, min_df=2, sublinear TF
- **Dataset:** UCI SMS Spam Collection (5,574 messages; 5,171 after removing exact duplicates)
- **Split:** 80/20 stratified, random_state=42
- **Test results (1,035 messages):** spam precision 0.930, spam recall 0.916, spam F1 0.923
- **Model choice:** Naive Bayes, Logistic Regression and Linear SVM were statistically
  tied in repeated cross-validation; Logistic Regression was chosen for its probability
  output and interpretability.
        """
    )
    st.warning("Limitation: the dataset is small and dates from around 2012, so "
               "modern spam styles may be missed. Not for production use.")

with tab_method:
    st.subheader("Methodology")
    st.markdown(
        """
1. **Data:** UCI SMS Spam Collection; exact duplicates removed before splitting to avoid leakage.
2. **Preprocessing:** lowercase, decode HTML entities, map URLs and number placeholders to
   tokens, keep digits and punctuation (strong spam signals).
3. **Features:** TF-IDF with 1-2 word n-grams.
4. **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM.
5. **Tuning:** GridSearchCV, 5-fold stratified CV on training data only, optimising spam F1.
6. **Evaluation:** held-out test set, precision/recall/F1, ROC-AUC and PR-AUC.
7. **Explanation:** word contribution = TF-IDF value x model coefficient.
        """
    )
