
import streamlit as st
import joblib
import re
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# Load saved model and vectorizer
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_resources()


# Clean email text
def clean_email(email):
    email = email.lower()
    email = re.sub(r'[^a-zA-Z\s]', '', email)
    email = re.sub(r'\s+', ' ', email).strip()
    return email


# Predict email
def predict_email(email):
    cleaned_email = clean_email(email)

    if not cleaned_email:
        return None

    email_features = vectorizer.transform([cleaned_email])
    prediction = model.predict(email_features)[0]

    return prediction


# Application interface
st.title("📧 Spam Email Detector")

st.write(
    "Enter an email message below to check "
    "whether it is Spam or Not Spam."
)

email_text = st.text_area(
    "Enter Email Text:",
    height=200,
    placeholder="Paste your email message here..."
)

if st.button("Predict Email", type="primary"):

    if not email_text.strip():
        st.warning("Please enter an email first.")

    else:
        result = predict_email(email_text)

        if result is None:
            st.warning("Please enter meaningful email text.")

        elif result == "spam":
            st.error("🚨 This email is SPAM!")

        else:
            st.success("✅ This email is NOT SPAM!")

st.caption(
    "Spam Email Classification using Naive Bayes and TF-IDF"
)
