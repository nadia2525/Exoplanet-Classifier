"""
Interactive Streamlit dashboard: predict whether a candidate is a
confirmed exoplanet or a false positive, and explore what drives the model.

Run: streamlit run app.py
"""
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="wide")

MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/koi_clean.csv"

FEATURE_INFO = {
    "koi_period": ("Orbital period (days)", 0.5, 500.0, 10.0),
    "koi_duration": ("Transit duration (hours)", 0.5, 20.0, 3.0),
    "koi_depth": ("Transit depth (ppm)", 10.0, 100000.0, 500.0),
    "koi_prad": ("Planetary radius (Earth radii)", 0.3, 30.0, 2.0),
    "koi_teq": ("Equilibrium temperature (K)", 100.0, 3000.0, 500.0),
    "koi_insol": ("Insolation flux (Earth = 1)", 0.01, 5000.0, 50.0),
    "koi_model_snr": ("Transit signal-to-noise ratio", 5.0, 500.0, 20.0),
    "koi_steff": ("Stellar effective temperature (K)", 3000.0, 10000.0, 5700.0),
    "koi_slogg": ("Stellar surface gravity (log g)", 3.0, 5.0, 4.4),
    "koi_srad": ("Stellar radius (Solar radii)", 0.1, 10.0, 1.0),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


bundle = load_model()
model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]
df = load_data()

st.title("🪐 Exoplanet Candidate Classifier")
st.write(
    "Predict whether a Kepler transit candidate is likely a **confirmed "
    "exoplanet** or a **false positive**, based on transit and stellar "
    "properties. Adjust the sliders to match a candidate's measurements."
)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Candidate properties")
    inputs = {}
    for col, (label, lo, hi, default) in FEATURE_INFO.items():
        inputs[col] = st.slider(label, lo, hi, default)

    X_input = pd.DataFrame([inputs])[feature_cols]
    X_scaled = scaler.transform(X_input)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = "CONFIRMED" if proba >= 0.5 else "FALSE POSITIVE"

    st.subheader("Prediction")
    st.metric("Predicted classification", pred)
    st.progress(float(proba))
    st.caption(f"Model confidence (probability of confirmed planet): {proba:.1%}")

with col2:
    st.subheader("What drives this model")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        importances.plot(kind="barh", ax=ax)
        ax.set_title("Feature importance")
        st.pyplot(fig)
    else:
        st.write("Feature importance not available for this model type.")

    st.subheader("Where your candidate sits vs. real data")
    feature_to_plot = st.selectbox("Compare on:", feature_cols, index=6)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for label_val, name, color in [(0, "False Positive", "tomato"), (1, "Confirmed", "steelblue")]:
        subset = df[df["label"] == label_val][feature_to_plot]
        ax2.hist(subset, bins=40, alpha=0.5, label=name, color=color)
    ax2.axvline(inputs[feature_to_plot], color="black", linestyle="--", label="Your input")
    ax2.set_xlabel(feature_to_plot)
    ax2.legend()
    st.pyplot(fig2)

st.divider()
st.caption(
    "Data: NASA Exoplanet Archive, Kepler Objects of Interest (Cumulative table). "
    "Model trained on confirmed vs. false-positive dispositions."
)
