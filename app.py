import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


    page_title="Ad Spend ROI Predictor",
    page_icon="📊",
    layout="centered"
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

CHANNEL_LABELS = [
    "Paid Views",
    "Google Impressions",
    "Email Impressions",
    "Facebook Impressions",
    "Affiliate Impressions",
]


st.title("📊 Ad Spend ROI Predictor")
st.markdown(
    "Enter your weekly impressions across each marketing channel "
    "and the app will predict your expected sales — and tell you "
    "which channel is actually pulling its weight."
)
st.markdown("---")

st.subheader("Weekly Channel Impressions")

col1, col2 = st.columns(2)

with col1:
    paid      = st.number_input("Paid Views",            min_value=0, value=5000, step=500)
    google    = st.number_input("Google Impressions",    min_value=0, value=3000, step=500)
    email     = st.number_input("Email Impressions",     min_value=0, value=2000, step=500)

with col2:
    facebook  = st.number_input("Facebook Impressions",  min_value=0, value=4000, step=500)
    affiliate = st.number_input("Affiliate Impressions", min_value=0, value=1000, step=500)

st.markdown("---")

# ── predict ───────────────────────────────────────────────────────────────────
if st.button(" Predict Sales & ROI", use_container_width=True):

    input_values  = [paid, google, email, facebook, affiliate]
    features      = np.array([input_values])
    predicted_sales = model.predict(features)[0]

    coefficients     = model.named_steps["model"].coef_
    contributions    = [imp * coef for imp, coef in zip(input_values, coefficients)]
    total            = sum(contributions) if sum(contributions) != 0 else 1
    contribution_pct = [round((c / total) * 100, 1) for c in contributions]

    best_channel  = CHANNEL_LABELS[int(np.argmax(contributions))]
    worst_channel = CHANNEL_LABELS[int(np.argmin(contributions))]

    # results
    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Sales",    f"{predicted_sales:,.0f} units")
    m2.metric("Best Channel 🏆",    best_channel)
    m3.metric("Underperforming ⚠️", worst_channel)

    st.markdown("---")

    # chart
    st.subheader("Channel Contribution to Sales")
    st.caption("How much each channel is driving sales based on your inputs.")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in contributions]
    bars   = ax.barh(CHANNEL_LABELS, contributions, color=colors)
    ax.set_xlabel("Sales Contribution (Impressions × Impact Score)")
    ax.set_title("Channel-wise Sales Contribution")
    ax.axvline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, contributions):
        ax.text(
            bar.get_width() + (max(contributions) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # table
    st.subheader("Channel Breakdown")

    def recommend(c):
        if c > 0 and c == max(contributions):
            return "✅ Scale this up"
        elif c > 0:
            return "✅ Keep investing"
        else:
            return "❌ Review strategy"

    summary_df = pd.DataFrame({
        "Channel":            CHANNEL_LABELS,
        "Impressions":        [f"{v:,}" for v in input_values],
        "Impact Score":       [round(c, 4) for c in coefficients],
        "Sales Contribution": [f"{c:,.0f}" for c in contributions],
        "Share of Sales %":   contribution_pct,
        "Recommendation":     [recommend(c) for c in contributions],
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Model: Ridge Regression | Trained on 3,051 weeks of real marketing data | CV R²: 0.657")
