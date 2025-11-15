import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to predict whether it is **Fraud** or **Not Fraud**.")

# -------------------------
# Load and train model
# -------------------------
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.subheader("📌 Enter Transaction Feature Values")

user_input = []

for col in X.columns:
    value = st.number_input(col, value=float(X[col].mean()))
    user_input.append(value)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input], columns=X.columns)

# Scale input SAME WAY as training data
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Fraud Transaction Detected!** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Legit (Probability of Fraud: {prob:.2f})")
