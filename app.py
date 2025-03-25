import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("sample_data.csv")

# Split features and target
X = df[["Age", "Salary"]]
y = df["Purchased"]

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# SHAP Explainer
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Streamlit UI
st.title("ML Model with SHAP Interpretability & Graphs 📊")

# User inputs
age = st.slider("Select Age:", min_value=18, max_value=100, value=30)
salary = st.slider("Select Salary:", min_value=20000, max_value=150000, value=50000)

# Predict button
if st.button("Predict"):
    user_data = scaler.transform([[age, salary]])
    prediction = model.predict(user_data)[0]
    
    st.subheader("🔹 Prediction Result")
    st.write(f"Prediction: **{'Purchased' if prediction == 1 else 'Not Purchased'}**")

    # SHAP Explanation
    shap_value = explainer(user_data)

    # SHAP Force Plot
    st.subheader("🔹 SHAP Force Plot")
    fig, ax = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_value.values, feature_names=["Age", "Salary"], matplotlib=True)
    st.pyplot(fig)

    # SHAP Summary Plot
    st.subheader("🔹 SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

    # SHAP Decision Plot
    st.subheader("🔹 SHAP Decision Plot")
    fig, ax = plt.subplots()
    shap.decision_plot(explainer.expected_value, shap_values.values, X_test)
    st.pyplot(fig)