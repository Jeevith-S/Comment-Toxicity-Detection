Comment Toxicity Detection 🔴
A deep learning system that classifies toxic comments across 6 categories using a PyTorch LSTM model, deployed as a real-time web application on Streamlit.

📌 What This Does
Given any comment or text input, this system predicts whether it is:
CategoryDescriptiontoxicGenerally harmful or offensivesevere_toxicExtremely offensive contentobsceneProfane or vulgar languagethreatContains threats of violenceinsultPersonal attacks or insultsidentity_hateHate speech targeting identity groups

🏆 Results
MetricValueValidation Accuracy~91%DatasetJigsaw Toxic Comment ClassificationModelPyTorch LSTMDeploymentStreamlit (real-time + bulk CSV)

🛠 Tech Stack

Model: PyTorch · LSTM (Long Short-Term Memory)
NLP: Text preprocessing · Tokenization · Embedding
Data: Jigsaw Toxic Comment dataset (Kaggle)
Deployment: Streamlit
Libraries: PyTorch · Pandas · NumPy · Scikit-learn


📁 Project Structure
Comment-Toxicity-Detection/
│
├── model/
│   ├── lstm_model.py          # LSTM architecture
│   └── trained_model.pt       # Saved model weights
│
├── app/
│   └── app.py                 # Streamlit application
│
├── data/
│   └── preprocessing.py       # Text cleaning pipeline
│
├── notebooks/
│   └── training.ipynb         # Model training notebook
│
├── requirements.txt
└── README.md

⚙️ How to Run Locally
bash# 1. Clone the repository
git clone https://github.com/Jeevith-S/Comment-Toxicity-Detection.git
cd Comment-Toxicity-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/app.py

🧠 Model Architecture
Input Text
    │
    ▼
Text Preprocessing (lowercase, remove special chars)
    │
    ▼
Tokenization & Padding
    │
    ▼
Embedding Layer (vocab_size × embedding_dim)
    │
    ▼
LSTM Layer (bidirectional, 2 layers)
    │
    ▼
Dropout (0.3)
    │
    ▼
Fully Connected Layer
    │
    ▼
Sigmoid Activation → 6 output scores (one per category)

📊 Features

Real-time prediction — type any comment and get instant toxicity scores
Bulk CSV prediction — upload a CSV file and get predictions for all rows
Visual confidence bars — see probability scores for each toxicity category
End-to-end pipeline — from raw text to final prediction in one click

