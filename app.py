import prob
import streamlit as st
import pandas as pd
import joblib

# Load trained model and top features
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

st.title("Telecom Churn Prediction")
st.write("Enter customer details to predict churn:")

# Example input fields (update based on top_features)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
Contract_Month_to_month = st.selectbox("Contract (Month-to-Month)", [0,1])

# Create a dataframe from input
input_data = pd.DataFrame({
    'MonthlyCharges': [MonthlyCharges],
    'tenure': [tenure],
    'Contract_Month-to-month': [Contract_Month_to_month],
})

# Add missing features
for col in top_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[top_features]

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    result = "Churn" if prediction[0]==1 else "No Churn"
    st.success(f"Prediction: {result}")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load trained model and top features
# -----------------------
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

st.title("📞 Telecom Churn Prediction App")
st.write("Predict whether a customer is likely to churn.")

# -----------------------
# Sidebar: Upload CSV for multiple predictions
# -----------------------
st.sidebar.title("Batch Prediction (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV for multiple customers", type=["csv"])
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

# -----------------------
# Single Customer Input
# -----------------------
st.subheader("Single Customer Input")

# Dynamically create input fields for numeric features
user_input = {}
for feature in top_features:
    # Example: if feature name contains 'Yes/No' or 'Contract' we can use selectbox
    if 'Yes' in feature or 'No' in feature or 'Contract' in feature or 'Payment' in feature:
        user_input[feature] = st.selectbox(feature, [0,1])
    else:
        user_input[feature] = st.number_input(feature, min_value=0.0, max_value=10000.0, value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df[top_features]  # Ensure correct order

# -----------------------
# Prediction Buttons
# -----------------------
if st.button("Predict Single Customer Churn"):
    prediction = model.predict(input_df)
    result = "Churn" if prediction[0]==1 else "No Churn"
    st.success(f"Prediction: {result}")

# Multiple Customer Prediction
if uploaded_file:
    st.subheader("Batch Predictions")
    # Fill missing columns if CSV lacks top_features
    for col in top_features:
        if col not in df_new.columns:
            df_new[col] = 0
    df_new = df_new[top_features]
    batch_pred = model.predict(df_new)
    df_new['Churn_Prediction'] = ["Churn" if p==1 else "No Churn" for p in batch_pred]
    st.dataframe(df_new)

# -----------------------
# Feature Importance
# -----------------------
st.subheader("Top 10 Feature Importance")
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': top_features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10))
plt.title("Top 10 Important Features")
st.pyplot(plt)

st.sidebar.title("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

col1, col2 = st.columns(2)
col1.number_input("Monthly Charges", ...)
col2.number_input("Tenure", ...)

with st.expander("Advanced Features"):
    st.number_input("Contract Length", ...)

if prob > 0.8:
    st.error("High Risk of Churn!")
elif prob > 0.5:
    st.warning("Medium Risk of Churn")
else:
    st.success("Low Risk of Churn")

st.progress(prob[0])
st.write(f"Churn Probability: {prob[0]*100:.2f}%")

st.markdown("""
**Instructions:**
- Fill all numeric fields carefully
- Use dropdowns for categorical inputs
- Click 'Predict' to see churn risk
""")

st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, help="Enter customer's monthly bill")

st.markdown("<h1 style='color: darkblue;'>Telecom Churn Predictor</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Customer", "Batch Prediction"])
with tab1:
    st.write("Single prediction here")
with tab2:
    st.write("Batch prediction upload here")

st.download_button("Download Predictions",df_new.to_csv(index=False), file_name="predictions.csv")

import streamlit as st

# Example layout
col1, col2 = st.columns(2)

# Monthly Charges
col1.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=10000.0,
    value=200.0,
    step=1.0
)

# Total Charges
col2.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=100000.0,
    value=5000.0,
    step=1.0
)

# Tenure in months
col1.number_input(
    "Tenure (Months)",
    min_value=0.0,
    max_value=120.0,
    value=12.0,
    step=1.0
)

# Any other numeric inputs
col2.number_input(
    "Some Other Numeric Value",
    min_value=0.0,
    max_value=1000.0,
    value=50.0,
    step=1.0
)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load trained model and top features
# -----------------------
model = joblib.load('randomforest_churn_model.pkl')
top_features = joblib.load('top_features.pkl')

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")
st.title("📞 Telecom Churn Prediction App")
st.markdown("""
Predict whether a customer is likely to churn.  
Fill the inputs below or upload a CSV file for batch prediction.
""")

# -----------------------
# Sidebar for Batch Upload
# -----------------------
st.sidebar.title("Batch Prediction (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

# -----------------------
# Tabs: Single Customer / Batch Prediction
# -----------------------
tab1, tab2 = st.tabs(["Single Customer", "Batch Prediction"])

# -----------------------
# Tab 1: Single Customer
# -----------------------
with tab1:
    st.subheader("Single Customer Input")
    user_input = {}

    # Split input fields into 2 columns
    col1, col2 = st.columns(2)
    for i, feature in enumerate(top_features):
        if 'Yes' in feature or 'No' in feature or 'Contract' in feature or 'Payment' in feature:
            val = st.selectbox(feature, [0, 1])
        else:
            val = st.number_input(feature, min_value=0.0, max_value=10000.0, value=0.0)
        if i % 2 == 0:
            user_input[feature] = val
        else:
            user_input[feature] = val

    input_df = pd.DataFrame([user_input])
    input_df = input_df[top_features]  # Ensure correct order

    if st.button("Predict Single Customer Churn"):
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)[:, 1][0]

        # Risk label
        if prob > 0.8:
            risk = "High Risk ⚠️"
            st.error(f"Prediction: {risk} (Probability: {prob * 100:.2f}%)")
        elif prob > 0.5:
            risk = "Medium Risk ⚠️"
            st.warning(f"Prediction: {risk} (Probability: {prob * 100:.2f}%)")
        else:
            risk = "Low Risk ✅"
            st.success(f"Prediction: {risk} (Probability: {prob * 100:.2f}%)")

# -----------------------
# Tab 2: Batch Prediction
# -----------------------
with tab2:
    if uploaded_file:
        st.subheader("Batch Predictions")
        # Fill missing top features
        for col in top_features:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new = df_new[top_features]

        batch_pred = model.predict(df_new)
        batch_prob = model.predict_proba(df_new)[:, 1]

        df_new['Churn_Prediction'] = ["Churn" if p == 1 else "No Churn" for p in batch_pred]
        df_new['Churn_Probability'] = batch_prob


        # Risk Label
        def risk_label(p):
            if p > 0.8:
                return "High Risk ⚠️"
            elif p > 0.5:
                return "Medium Risk ⚠️"
            else:
                return "Low Risk ✅"


        df_new['Risk_Level'] = df_new['Churn_Probability'].apply(risk_label)

        st.dataframe(df_new)

        # Summary
        st.markdown("**Summary of Batch Predictions:**")
        st.write(df_new['Risk_Level'].value_counts())

        # Download button
        csv = df_new.to_csv(index=False)
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file in the sidebar to enable batch prediction.")

# -----------------------
# Feature Importance Chart
# -----------------------
st.subheader("Top 10 Feature Importance")
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': top_features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10))
plt.title("Top 10 Important Features")
st.pyplot(plt)

def churn_recommendation(prob):
    if prob > 0.8:
        return [
            "⚠️ Offer discount to retain customer",
            "⚠️ Assign priority customer support",
            "⚠️ Check if contract plan can be upgraded"
        ]
    elif prob > 0.5:
        return [
            "⚠️ Monitor customer usage",
            "⚠️ Send engagement emails or offers"
        ]
    else:
        return ["✅ Customer seems low-risk"]

