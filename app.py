import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("📸 Instagram Fake Account Detector")

# Load trained model
model = joblib.load("outputs/best_model.pkl")

st.write("Upload a CSV file with account data to check predictions:")

uploaded_file = st.file_uploader("Choose a CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # ---- Feature engineering (same as training pipeline) ----
    for c in ['posts','followers','follows']:
        data[f'log_{c}'] = np.log1p(data[c].astype(float))

    data['foll_ratio'] = data['followers'] / (data['follows'] + 1)

    for b in ['profile_pic','external_url','private','name__username']:
        data[f'{b}_int'] = data[b].astype(int)

    data['desc_per_word'] = data['description_length'] / (data['fullname_words'] + 1)

    features = [c for c in data.columns if data[c].dtype in [int,float] and c != 'fake']
    X = data[features].fillna(0)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    result = pd.DataFrame({
        "id": data.index,
        "fake_prediction": preds,
        "fake_probability": probs
    })

    st.subheader("🔮 Predictions")
    st.write(result)

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")
