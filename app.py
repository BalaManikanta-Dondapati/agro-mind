import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AgroMind AI", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("models/crop_health_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# -------------------------------
# LOAD SENTIMENT DATA
# -------------------------------
sentiment_df = pd.read_csv("data/social_media/sentiment_text.csv")

# -------------------------------
# TITLE
# -------------------------------
st.title("🌿 AgroMind: Crop Mood Estimation")
st.write("AI-based Crop Health + Farmer Sentiment Analysis")

# -------------------------------
# INPUT
# -------------------------------
st.subheader("📥 Input")

uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg","png","jpeg"])

mode = st.radio("Sentiment Mode", ["Manual", "Dataset"])

if mode == "Manual":
    text = st.text_input("Enter farmer statement")
    sentiment = st.selectbox("Select Sentiment", [-1, 0, 1])
else:
    idx = st.slider("Select Dataset Sample", 0, len(sentiment_df)-1, 0)
    text = sentiment_df.iloc[idx]["text"]
    sentiment = int(sentiment_df.iloc[idx]["label"])   # ✅ FIXED
    st.info(f"💬 {text}")

# -------------------------------
# IMAGE FEATURE EXTRACTION
# -------------------------------
def extract_features(image):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (128,128))

    red = img[:,:,2].astype(float)
    green = img[:,:,1].astype(float)

    ndvi = (green - red) / (green + red + 1e-5)

    return {
        "ndvi_mean": np.mean(ndvi),
        "ndvi_std": np.std(ndvi),
        "ndvi_min": np.min(ndvi),
        "ndvi_max": np.max(ndvi),
        "green_mean": np.mean(green)/255,
        "red_mean": np.mean(red)/255,
        "ndvi_map": ndvi
    }

# -------------------------------
# PREDICT
# -------------------------------
if st.button("🚀 Predict"):

    if uploaded_file is None:
        st.warning("Please upload image")
    else:
        features = extract_features(uploaded_file)

        # CMI
        cmi = 0
        if features["ndvi_mean"] > 0.5:
            cmi += 2
        elif features["ndvi_mean"] > 0.2:
            cmi += 1
        else:
            cmi -= 1
        cmi += sentiment

        # MODEL INPUT
        X = np.array([[features["ndvi_mean"], features["ndvi_std"],
                       features["ndvi_min"], features["ndvi_max"],
                       features["green_mean"], features["red_mean"],
                       sentiment, cmi]])

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        label = le.inverse_transform(pred)[0]
        confidence = np.max(probs) * 100

        # COLOR
        if label == "healthy":
            color = "#22c55e"
            insight = "🌱 Excellent crop condition"
        elif label == "moderate":
            color = "#facc15"
            insight = "⚖️ Crop needs monitoring"
        else:
            color = "#ef4444"
            insight = "⚠️ Crop stress detected"

        # -------------------------------
        # IMAGE
        # -------------------------------
        st.image(uploaded_file, use_container_width=True)

        # -------------------------------
        # RESULT
        # -------------------------------
        st.subheader("🌱 Prediction Result")
        st.markdown(f"### {label.upper()}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # -------------------------------
        # INPUTS
        # -------------------------------
        st.subheader("🔍 Model Inputs")
        st.write(f"NDVI Mean: {features['ndvi_mean']:.3f}")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"CMI Score: {cmi}")

        # -------------------------------
        # INSIGHTS
        # -------------------------------
        st.subheader("🧠 Insights")
        st.success(insight)

        # -------------------------------
        # VISUAL INSIGHTS
        # -------------------------------
        st.markdown("---")
        st.subheader("📊 Visual Insights")

        c1, c2, c3 = st.columns([1,2,1])

        with c2:
            # Confidence Line
            fig, ax = plt.subplots(figsize=(4,2))
            ax.plot([0, confidence], [0, 0], linewidth=6, color=color, marker='o')
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_title("Confidence (%)", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
            st.pyplot(fig)

            # NDVI Line
            fig, ax = plt.subplots(figsize=(4,2))
            ax.plot([0, features["ndvi_mean"]], [0, 0], linewidth=6, color=color, marker='o')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_title("NDVI Level", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
            st.pyplot(fig)

        # -------------------------------
        # NDVI HEATMAP
        # -------------------------------
        st.subheader("🌈 NDVI Heatmap")

        c1, c2, c3 = st.columns([1,2,1])

        with c2:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(features["ndvi_map"], cmap="RdYlGn")
            ax.axis("off")
            st.pyplot(fig)

        # -------------------------------
        # EXPLAINABLE AI
        # -------------------------------
        st.markdown("---")
        st.subheader("🧠 Explainable AI (Feature Importance)")

        feature_names = [
            "NDVI Mean","NDVI Std","NDVI Min","NDVI Max",
            "Green Mean","Red Mean","Sentiment","CMI"
        ]

        importances = model.feature_importances_
        importances = importances / np.sum(importances)

        sorted_idx = np.argsort(importances)

        c1, c2, c3 = st.columns([1,2,1])

        with c2:
            fig, ax = plt.subplots(figsize=(5,3))

            colors = ["#22c55e" if f in ["NDVI Mean","CMI"] else "#38bdf8"
                      for f in feature_names]

            ax.barh(np.array(feature_names)[sorted_idx],
                    importances[sorted_idx],
                    color=np.array(colors)[sorted_idx])

            ax.set_title("Feature Contribution", fontsize=10)

            for s in ax.spines.values():
                s.set_visible(False)

            st.pyplot(fig)