import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Fake Review Detector", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 20px;
        border-radius: 15px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #f0f2f6;
        color: black;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🔍 Fake Review Detection System</h1>", unsafe_allow_html=True)
st.markdown("### Detect whether a review is **Fake or Real** using Machine Learning 🚀")

# ---------- LOAD DATA ----------
df = pd.read_csv("reviews.csv")

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# ---------- NLP ----------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['label']

# ---------- MODEL ----------
model = LogisticRegression()
model.fit(X, y)

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- LEFT SIDE ----------
with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Review Distribution")
    fig, ax = plt.subplots()
    df['label'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ---------- RIGHT SIDE ----------
with col2:
    st.subheader("📝 Enter Your Review")

    user_input = st.text_area("Type your review here...")

    if st.button("🔍 Predict"):
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)

        if prediction[0] == "real":
            st.success("✅ This is a REAL Review")
        else:
            st.error("❌ This is a FAKE Review")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Made with ❤️ using Python & Machine Learning")


import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Fake Review Detection", layout="centered")

# Title
st.title("🕵️ Fake Review Detection System")

# Description
st.write("This application detects whether a review is genuine or fake using basic NLP logic.")

# Upload file
uploaded_file = st.file_uploader("📂 Upload Review Dataset (CSV)")

if uploaded_file is not None:
    data= pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("Total Reviews:", len(data))

    st.subheader("📊 Review Distribution")
    if 'label' in data.columns:
        st.bar_chart(data['label'].value_counts())  

    st.subheader("📈 Dataset Statistics")
    st.write(data.describe())

# Divider
st.markdown("---")

# User input
st.subheader("✍️ Enter Review for Prediction")
review = st.text_area("Type your review here:")

if st.button("🔍 Check Review"):
    if review:
        # Improved dummy logic
        if any(word in review.lower() for word in ["good", "great", "excellent", "amazing"]):
            st.success("✅ This review looks Genuine")
        else:
            st.error("⚠️ This review might be Fake")
    else:
        st.warning("Please enter a review")

# Footer
st.markdown("---")
st.caption("Developed by Chaitali Vahadane")