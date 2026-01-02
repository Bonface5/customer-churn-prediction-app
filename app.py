import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.caption("Predict customer churn risk using a trained Machine Learning model")
st.divider()

# =============================
# Load Model & Preprocessor
# =============================
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# =============================
# User Input Section
# =============================
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=60, value=24)
    usage = st.number_input("Usage Frequency", min_value=0.0, max_value=100.0, value=30.0)
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=0)

with col2:
    payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=0)
    total_spend = st.number_input("Total Spend", min_value=0.0, value=2500.0)
    last_interaction = st.number_input("Last Interaction (days ago)", min_value=1, max_value=30, value=2)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], index=2)
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"], index=2)

st.divider()

# =============================
# Prediction
# =============================
if st.button("🔍 Predict Churn"):

    # -----------------------------
    # Create input DataFrame
    # -----------------------------
    input_data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "Usage Frequency": [usage],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction],
        "Gender": [gender],
        "Subscription Type": [subscription],
        "Contract Length": [contract]
    })

    # -----------------------------
    # Feature Engineering (must match training)
    # -----------------------------
    input_data["Late_Payer"] = (input_data["Payment Delay"] > 0).astype(int)
    input_data["High_Support_User"] = (input_data["Support Calls"] >= 3).astype(int)
    input_data["Low_Usage_User"] = (input_data["Usage Frequency"] < 10).astype(int)

    # -----------------------------
    # Preprocess & Predict
    # -----------------------------
    input_processed = preprocessor.transform(input_data)
    churn_prob = model.predict_proba(input_processed)[0][1]

    # =============================
    # Results Section
    # =============================
    st.subheader("📊 Prediction Result")

    # 1️⃣ Metric
    st.metric("Churn Probability", f"{churn_prob:.2%}")

    # 2️⃣ Visual Risk Indicator
    st.progress(min(churn_prob, 1.0))

    # 3️⃣ Risk Classification
    if churn_prob >= 0.7:
        risk_label = "🔴 High Risk"
        st.error("Immediate retention action recommended")
    elif churn_prob >= 0.5:
        risk_label = "🟡 Medium Risk"
        st.warning("Customer should be monitored and engaged")
    else:
        risk_label = "🟢 Low Risk"
        st.success("Customer is likely to stay")

    st.write(f"**Risk Category:** {risk_label}")

    # =============================
    # Interpretation
    # =============================
    st.write("### How to interpret this score")
    st.markdown("""
    - **Below 50%** → Low churn risk  
    - **50% – 69%** → Medium risk (monitor & engage)  
    - **70% and above** → High risk (retention action required)
    """)

    # =============================
    # Business Insight
    # =============================
    st.write("### What this means for the business")
    st.markdown("""
    - High churn risk customers may require **discounts, loyalty offers, or follow-up calls**
    - Medium risk customers benefit from **engagement campaigns**
    - Low risk customers are good candidates for **upselling or referrals**
    """)

    # =============================
    # Feature Importance Visualization
    # =============================
    st.divider()
    st.subheader("🔍 Top Factors Influencing Churn")

    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Important Features")

        st.pyplot(fig)

    except Exception:
        st.info("Feature importance visualization not available for this model.")
