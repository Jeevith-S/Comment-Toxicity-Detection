%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# Text Cleaning Function
# -----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------------
# LSTM Model
# -----------------------------------
class ToxicityLSTM(nn.Module):
    def __init__(self):
        super(ToxicityLSTM, self).__init__()
        self.embedding = nn.Embedding(5000, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# -----------------------------------
# Load Assets
# -----------------------------------
@st.cache_resource
def load_assets():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = ToxicityLSTM()
    model.load_state_dict(torch.load("toxicity_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_assets()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.title("Comment Toxicity Detector")

# --- Single Prediction ---
st.header("Single Comment Prediction")
comment = st.text_area("Enter Comment")

if st.button("Analyze"):
    if comment.strip():
        cleaned = clean_text(comment)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200)
        input_t = torch.tensor(padded, dtype=torch.long)
        with torch.no_grad():
            out = model(input_t)
            probs = torch.sigmoid(out).numpy()[0]

        st.subheader("Results")
        cols = st.columns(3)
        for i, label in enumerate(labels):
            with cols[i % 3]:
                st.metric(label, f"{probs[i]:.2%}")
                st.progress(float(probs[i]))
    else:
        st.warning("Please enter text.")

# --- Bulk Prediction ---
st.header("Bulk CSV Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment_text" not in df.columns:
        st.error("CSV must have a 'comment_text' column.")
    else:
        with st.spinner('Analyzing...'):
            all_probs = []
            for text in df["comment_text"]:
                cleaned = clean_text(text)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=200)
                input_t = torch.tensor(padded, dtype=torch.long)
                with torch.no_grad():
                    out = model(input_t)
                    probs = torch.sigmoid(out).numpy()[0]
                    all_probs.append(probs)

            prob_array = np.array(all_probs)
            for i, label in enumerate(labels):
                df[label] = (prob_array[:, i] > 0.5).astype(int)

            st.success("Analysis Complete!")
            st.dataframe(df.head())
            st.download_button("Download CSV with Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")