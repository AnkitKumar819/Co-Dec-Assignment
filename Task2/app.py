#type:ignore
import streamlit as st
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------- Text Cleaning ----------------
sws = list(ENGLISH_STOP_WORDS)
for stop in ["not", "no"]:
    if stop in sws:
        sws.remove(stop)
sws = [w for w in sws if not w.endswith("n't")]

def text_cleaning(doc):
    newdoc = re.sub(r'[^a-z0-9\s]', '', doc.lower())
    words = newdoc.split()
    newdoc = [word for word in words if word not in sws]
    return ' '.join(newdoc)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Movie Sentiment Classifier", page_icon="💬", layout="centered")

st.title("💬Movie Sentiment Analysis App")
st.write("Enter a movie's review below to check if it's **Positive** or **Negative**.")

# Input box
review_text = st.text_area("Enter Review:", height=150)

if st.button("Movie Predict Sentiment"):
    if review_text.strip():
        cleaned_text = text_cleaning(review_text)
        prediction = model.predict([cleaned_text])[0]
        if prediction == 1:
            st.success("✅ Positive Sentiment (Liked)")
        else:
            st.error("❌ Negative Sentiment (Not Liked)")
    else:
        st.warning("⚠️ Please enter a review text before predicting.")

# Footer
st.caption("Model: CountVectorizer (binary=True) + BernoulliNB | Trained on reviews_data.csv")
