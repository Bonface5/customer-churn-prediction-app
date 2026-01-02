import streamlit as st
import pandas as pd
import pickle
import altair as alt

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

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("Prediction Result")
    st.metric(label="Churn Probability", value=f"{churn_prob:.2%}")

    # Tuned thresholds for Low / Medium / High
    if churn_prob >= 0.7:
        st.error("🔴 High risk of churn – immediate retention action recommended")
    elif churn_prob >= 0.5:
        st.warning("🟡 Medium risk of churn – monitor closely")
    else:
        st.success("🟢 Low risk of churn – customer likely to stay")

    # -----------------------------
    # Probability Interpretation
    # -----------------------------
    st.write("### How to interpret this score")
    st.markdown("""
    - **Below 50%** → Low churn risk  
    - **50% – 69%** → Medium risk (monitor & engage)  
    - **70% and above** → High risk (retention action required)
    """)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.divider()
    st.subheader("Top Factors Influencing Churn")

    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Display table
        st.write("Top 10 most important features used by the model:")
        st.dataframe(importance_df.head(10), use_container_width=True)

        # Top 5 bar chart
        st.write("### Visualizing Top 5 Features")
        top5 = importance_df.head(5).sort_values(by="Importance")  # ascending for horizontal bars

        chart = alt.Chart(top5).mark_bar(color='#FF6F61').encode(
            x=alt.X('Importance', title='Importance Score'),
            y=alt.Y('Feature', sort=None, title='Feature')
        ).properties(height=250, width=500)

        st.altair_chart(chart, use_container_width=True)

    except Exception:
        st.info("Feature importance visualization not available for this model setup.")
