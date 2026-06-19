import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("bilstm_toxicity.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Labels
# -----------------------------
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# -----------------------------
# Prediction Function
# -----------------------------
def predict_toxicity(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=200
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0]

    return prediction


# -----------------------------
# Page Title
# -----------------------------
st.title("💬 Comment Toxicity Detection")

st.write(
    """
    This project detects toxic comments using a Deep Learning BiLSTM model.
    """
)

# -----------------------------
# Project Overview
# -----------------------------
st.header("📌 Project Overview")

st.write("""
Online platforms receive millions of comments every day.
This model helps identify toxic comments automatically.

Classes:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate
""")

# -----------------------------
# Model Performance
# -----------------------------
st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        "99.41%"
    )

    st.metric(
        "Precision",
        "81.85%"
    )

with col2:
    st.metric(
        "Recall",
        "63.16%"
    )

    st.metric(
        "Macro F1",
        "42.85%"
    )

# -----------------------------
# Single Comment Prediction
# -----------------------------
st.header("✍️ Single Comment Prediction")

user_text = st.text_area(
    "Enter Comment"
)

if st.button("Predict"):

    if user_text.strip() != "":

        probs = predict_toxicity(
            user_text
        )

        st.subheader(
            "Prediction Results"
        )

        for label, score in zip(
            labels,
            probs
        ):

            st.write(
                f"**{label}** : {score*100:.2f}%"
            )

# -----------------------------
# CSV Upload
# -----------------------------
st.header("📂 Bulk Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(
        uploaded_file
    )

    if "comment_text" not in df.columns:

        st.error(
            "CSV must contain comment_text column"
        )

    else:

        predictions = []

        for text in df["comment_text"]:

            probs = predict_toxicity(
                str(text)
            )

            predictions.append(
                probs
            )

        pred_df = pd.DataFrame(
            predictions,
            columns=labels
        )

        result = pd.concat(
            [df, pred_df],
            axis=1
        )

        st.dataframe(
            result.head()
        )

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "tox_predictions.csv",
            "text/csv"
        )